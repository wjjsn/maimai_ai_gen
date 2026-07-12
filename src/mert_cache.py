"""使用冻结的 Hugging Face MERT 提取并缓存整曲时序特征。"""

import concurrent.futures
from collections import defaultdict, deque
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import transformers
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

from config import CONFIG


MODEL_DIR = CONFIG.mert.model_dir
CACHE_DIR = CONFIG.paths.mert_cache_dir
CACHE_VERSION = 3


def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def _state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _model_signature(model_dir: Path = MODEL_DIR) -> dict[str, dict[str, int]]:
    names = sorted(
        path.name
        for path in model_dir.iterdir()
        if path.is_file() and (
            path.suffix in (".json", ".py", ".bin", ".safetensors")
            or ".bin." in path.name
            or ".safetensors." in path.name
        )
    )
    return {name: _state(model_dir / name) for name in names}


def feature_signature() -> dict:
    """描述会改变冻结 MERT 特征数值的实现和配置。"""
    return {
        "cache_version": CACHE_VERSION,
        "model": _model_signature(),
        "chunk_sec": CONFIG.mert.chunk_sec,
        "context_sec": CONFIG.mert.context_sec,
        "batch_size": CONFIG.mert.batch_size,
        "float16": CONFIG.mert.cache_float16,
        "python": sys.version,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }


def load_mert(device: torch.device):
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    model.requires_grad_(False)
    if model.config.hidden_size != CONFIG.model.state:
        raise ValueError(
            f"MERT 输出维度为 {model.config.hidden_size}，配置状态维度为 {CONFIG.model.state}"
        )
    return processor, model


def _output_length(model, samples: int) -> int:
    return int(model._get_feat_extract_output_lengths(torch.tensor(samples)).item())


def _feature_centers(model, count: int, input_start: int, device: torch.device) -> torch.Tensor:
    stride = 1
    receptive_field = 1
    for kernel, layer_stride in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive_field += (kernel - 1) * stride
        stride *= layer_stride
    return input_start + torch.arange(count, device=device) * stride + (receptive_field - 1) / 2


def _prepare_waveform(waveform: torch.Tensor, source_rate: int, sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    waveform = waveform.float()
    return (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)


def _load_prepared_audio(path: str, sample_rate: int) -> tuple[np.ndarray, float]:
    """在 CPU 工作线程中完成解码和重采样，主进程独占 CUDA。"""
    started = time.perf_counter()
    try:
        waveform, source_rate = torchaudio.load(path)
    except RuntimeError as torchcodec_error:
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", path,
                "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
        if samples.size == 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"音频解码失败: {path}: {detail}") from torchcodec_error
        tqdm.write(f"[mert-cache] 警告: TorchCodec 解码失败，使用 FFmpeg 恢复: {path}")
        waveform = torch.from_numpy(samples)
        source_rate = sample_rate
    return _prepare_waveform(waveform, source_rate, sample_rate).numpy(), time.perf_counter() - started


def _chunks(waveform: torch.Tensor, model, sample_rate: int) -> list[tuple[torch.Tensor, int, int, int]]:
    core_samples = round(CONFIG.mert.chunk_sec * sample_rate)
    context_samples = round(CONFIG.mert.context_sec * sample_rate)
    stride = int(np.prod(model.config.conv_stride))
    chunks = []
    for core_start in range(0, waveform.numel(), core_samples):
        core_end = min(waveform.numel(), core_start + core_samples)
        input_start = max(0, core_start - context_samples)
        input_start -= input_start % stride
        input_end = min(waveform.numel(), core_end + context_samples)
        chunks.append((waveform[input_start:input_end], input_start, core_start, core_end))
    return chunks


@torch.inference_mode()
def _extract_chunks(chunks: list[tuple[torch.Tensor, int, int, int]], model, device: torch.device, batch_size: int) -> np.ndarray:
    """仅合并等长 chunk，确保与逐条推理的边界语义完全一致。"""
    grouped: dict[int, list[tuple[int, torch.Tensor, int, int, int]]] = defaultdict(list)
    for index, (chunk, input_start, core_start, core_end) in enumerate(chunks):
        grouped[chunk.numel()].append((index, chunk, input_start, core_start, core_end))
    pieces: list[torch.Tensor | None] = [None] * len(chunks)
    centers_by_piece: list[torch.Tensor | None] = [None] * len(chunks)
    for group in grouped.values():
        offset = 0
        while offset < len(group):
            current_size = min(batch_size, len(group) - offset)
            while True:
                current = group[offset:offset + current_size]
                batch = torch.stack([item[1] for item in current]).to(device)
                try:
                    hidden = model(input_values=batch).last_hidden_state
                    break
                except torch.OutOfMemoryError:
                    del batch
                    torch.cuda.empty_cache()
                    if current_size == 1:
                        raise
                    current_size //= 2
                    tqdm.write(f"[mert-cache] 警告: 显存不足，MERT batch 降至 {current_size}")
            hidden_cpu = hidden.cpu()
            for row, (index, _chunk, input_start, core_start, core_end) in enumerate(current):
                centers = _feature_centers(model, hidden.shape[1], input_start, device)
                keep = (centers >= core_start) & (centers < core_end)
                pieces[index] = hidden_cpu[row, keep.cpu()]
                centers_by_piece[index] = centers[keep].cpu()
            offset += current_size
    centers = torch.cat([item for item in centers_by_piece if item is not None])
    if centers.numel() > 1 and not torch.all(centers[1:] > centers[:-1]):
        raise RuntimeError("MERT 分块拼接后的特征中心不严格递增")
    return torch.cat([item for item in pieces if item is not None]).numpy()


@torch.inference_mode()
def extract_waveform_features(
    waveform: torch.Tensor,
    source_rate: int,
    device: torch.device,
    processor=None,
    model=None,
) -> np.ndarray:
    processor, model = (processor, model) if processor is not None else load_mert(device)
    waveform = _prepare_waveform(waveform, source_rate, processor.sampling_rate)
    features = _extract_chunks(
        _chunks(waveform, model, processor.sampling_rate), model, device, CONFIG.mert.batch_size
    )
    dtype = np.float16 if CONFIG.mert.cache_float16 else np.float32
    return features.astype(dtype, copy=False)


def extract_audio_features(path: str | Path, device: torch.device, processor=None, model=None) -> np.ndarray:
    path = Path(path)
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except RuntimeError as torchcodec_error:
        # 部分 MP3 只有个别损坏 packet；FFmpeg 可跳过坏包并恢复其余有效音频。
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-f", "f32le", "-ac", "1", "-ar", "24000", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
        if samples.size == 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"音频解码失败: {path}: {detail}") from torchcodec_error
        print(f"[mert-cache] TorchCodec 解码失败，使用 FFmpeg 恢复: {path}")
        waveform = torch.from_numpy(samples).unsqueeze(0)
        sample_rate = 24000
    return extract_waveform_features(waveform, sample_rate, device, processor, model)


def _expected(track: Path, signature: dict, sample_rate: int, hidden_size: int) -> dict:
    return {
        **signature,
        "track": _state(track),
        "sample_rate": sample_rate,
        "hidden_size": hidden_size,
    }


def _cache_is_current(out: Path, expected: dict) -> bool:
    try:
        return out.exists() and json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_cache(out: Path, features: np.ndarray, expected: dict) -> None:
    metadata_path = out.with_suffix(".json")
    temp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with temp.open("wb") as file:
        np.save(file, features)
    temp.replace(out)
    meta_temp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    meta_temp.write_text(json.dumps(expected, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    meta_temp.replace(metadata_path)


def process_one(chart_dir: Path, rel_path: str, cache_dir: Path, device, processor, model) -> tuple[Path, bool]:
    cache_dir.mkdir(exist_ok=True, parents=True)
    out = cache_dir / f"{cache_key(rel_path)}.npy"
    track = chart_dir / "track.mp3"
    expected = _expected(track, feature_signature(), processor.sampling_rate, model.config.hidden_size)
    if _cache_is_current(out, expected):
        return out, False
    features = extract_audio_features(track, device, processor, model)
    _write_cache(out, features, expected)
    return out, True


def main(charts_dir=CONFIG.paths.charts_dir, cache_dir=CACHE_DIR):
    charts_dir, cache_dir = Path(charts_dir), Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    signature = feature_signature()
    pending = []
    for chart_dir, _dirs, files in charts_dir.walk():
        if "track.mp3" not in files or "maidata.txt" not in files:
            continue
        rel = str(chart_dir.relative_to(charts_dir))
        out = cache_dir / f"{cache_key(rel)}.npy"
        expected = _expected(chart_dir / "track.mp3", signature, 24000, CONFIG.model.state)
        if not _cache_is_current(out, expected):
            pending.append((chart_dir, rel, out, expected))
    if not pending:
        print("[mert-cache] 全部命中，无需加载 MERT 模型")
        return 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_mert(device)
    if processor.sampling_rate != 24000 or model.config.hidden_size != CONFIG.model.state:
        raise RuntimeError("MERT 运行配置与缓存预检查不一致")
    count = 0
    skipped = 0
    audio_seconds = 0.0
    gpu_seconds = 0.0
    write_seconds = 0.0
    started = time.perf_counter()
    workers = CONFIG.mert.audio_workers
    print(f"[mert-cache] 重建 {len(pending)} 首，CPU workers={workers}，MERT batch<={CONFIG.mert.batch_size}")
    tasks = iter(pending)
    futures: dict[concurrent.futures.Future, tuple[Path, str, Path, dict]] = {}
    ready: deque[tuple[Path, str, Path, dict, np.ndarray]] = deque()
    # torchaudio 的解码和重采样在原生代码中执行；线程可与单卡推理并行，
    # 也不会因 train.py 的模块级入口被 spawn 子进程重复执行。
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
        total=len(pending), desc="[mert-cache] 重建", unit="首"
    ) as progress:
        def submit_available() -> None:
            while len(futures) < workers * 2:
                try:
                    chart_dir, rel, out, expected = next(tasks)
                except StopIteration:
                    return
                future = executor.submit(_load_prepared_audio, str(chart_dir / "track.mp3"), processor.sampling_rate)
                futures[future] = (chart_dir, rel, out, expected)

        submit_available()
        while futures or ready:
            if not ready:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    chart_dir, rel, out, expected = futures.pop(future)
                    try:
                        samples, elapsed = future.result()
                        audio_seconds += elapsed
                        ready.append((chart_dir, rel, out, expected, samples))
                    except Exception as error:
                        skipped += 1
                        progress.update()
                        tqdm.write(f"[mert-cache] 错误: 跳过 {rel}: {error}")
                submit_available()
                continue
            chart_dir, rel, out, expected, samples = ready.popleft()
            try:
                waveform = torch.from_numpy(samples)
                gpu_started = time.perf_counter()
                features = _extract_chunks(
                    _chunks(waveform, model, processor.sampling_rate), model, device, CONFIG.mert.batch_size
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                dtype = np.float16 if CONFIG.mert.cache_float16 else np.float32
                write_started = time.perf_counter()
                _write_cache(out, features.astype(dtype, copy=False), expected)
                write_seconds += time.perf_counter() - write_started
                count += 1
                progress.update()
            except Exception as error:
                skipped += 1
                progress.update()
                tqdm.write(f"[mert-cache] 错误: 跳过 {rel}: {error}")

    elapsed = time.perf_counter() - started
    print(
        f"[mert-cache] 完成: {count} 首，跳过 {skipped} 首，用时 {elapsed:.1f}s；"
        f"CPU音频累计 {audio_seconds:.1f}s，GPU累计 {gpu_seconds:.1f}s，写入累计 {write_seconds:.1f}s"
    )
    return count, skipped


def _self_check() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_mert(device)
    assert processor.sampling_rate == 24000
    assert model.config.hidden_size == 768
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    features = extract_waveform_features(torch.zeros(1, processor.sampling_rate), processor.sampling_rate, device, processor, model)
    assert features.ndim == 2 and features.shape[1] == 768
    assert np.isfinite(features).all()
    assert _output_length(model, processor.sampling_rate) == 74
    # 相同长度 batch 只允许 GPU 浮点归约顺序造成的微小差异。
    samples = torch.randn(2, processor.sampling_rate, device=device)
    batched = model(input_values=samples).last_hidden_state
    single = model(input_values=samples[:1]).last_hidden_state
    assert (batched[:1] - single).abs().max() < 0.003
    print(f"[mert-cache] 自检通过: {features.shape}")


if __name__ == "__main__":
    _self_check()
