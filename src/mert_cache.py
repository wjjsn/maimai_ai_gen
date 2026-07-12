"""使用冻结的 Hugging Face MERT 提取并缓存整曲时序特征。"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from config import CONFIG


MODEL_DIR = CONFIG.mert.model_dir
CACHE_DIR = CONFIG.paths.mel_cache_dir
CACHE_VERSION = 1


def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def _state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _model_signature(model_dir: Path = MODEL_DIR) -> dict[str, dict[str, int]]:
    names = ("config.json", "preprocessor_config.json", "modeling_MERT.py", "configuration_MERT.py", "pytorch_model.bin")
    return {name: _state(model_dir / name) for name in names}


def load_mert(device: torch.device):
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    model.requires_grad_(False)
    if model.config.hidden_size != CONFIG.model.audio_state:
        raise ValueError(
            f"MERT 输出维度为 {model.config.hidden_size}，配置音频状态维度为 {CONFIG.model.audio_state}"
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


@torch.inference_mode()
def extract_waveform_features(
    waveform: torch.Tensor,
    source_rate: int,
    device: torch.device,
    processor=None,
    model=None,
) -> np.ndarray:
    processor, model = (processor, model) if processor is not None else load_mert(device)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if source_rate != processor.sampling_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, processor.sampling_rate)
    waveform = waveform.float()
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)

    rate = processor.sampling_rate
    core_samples = round(CONFIG.mert.chunk_sec * rate)
    context_samples = round(CONFIG.mert.context_sec * rate)
    stride = int(np.prod(model.config.conv_stride))
    total_samples = waveform.numel()
    pieces = []
    for core_start in range(0, total_samples, core_samples):
        core_end = min(total_samples, core_start + core_samples)
        input_start = max(0, core_start - context_samples)
        input_start -= input_start % stride
        input_end = min(total_samples, core_end + context_samples)
        chunk = waveform[input_start:input_end].unsqueeze(0).to(device)
        hidden = model(input_values=chunk).last_hidden_state[0]
        centers = _feature_centers(model, hidden.shape[0], input_start, device)
        keep = (centers >= core_start) & (centers < core_end)
        pieces.append(hidden[keep].cpu())
    features = torch.cat(pieces).numpy()
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


def process_one(chart_dir: Path, rel_path: str, cache_dir: Path, device, processor, model) -> tuple[Path, bool]:
    cache_dir.mkdir(exist_ok=True, parents=True)
    out = cache_dir / f"{cache_key(rel_path)}.npy"
    metadata_path = out.with_suffix(".json")
    track = chart_dir / "track.mp3"
    expected = {
        "cache_version": CACHE_VERSION,
        "track": _state(track),
        "model": _model_signature(),
        "sample_rate": processor.sampling_rate,
        "hidden_size": model.config.hidden_size,
        "chunk_sec": CONFIG.mert.chunk_sec,
        "context_sec": CONFIG.mert.context_sec,
        "float16": CONFIG.mert.cache_float16,
    }
    try:
        if out.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) == expected:
            return out, False
    except (OSError, json.JSONDecodeError):
        pass
    features = extract_audio_features(track, device, processor, model)
    temp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with temp.open("wb") as file:
        np.save(file, features)
    temp.replace(out)
    meta_temp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    meta_temp.write_text(json.dumps(expected, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    meta_temp.replace(metadata_path)
    return out, True


def main(charts_dir=CONFIG.paths.charts_dir, cache_dir=CACHE_DIR):
    charts_dir, cache_dir = Path(charts_dir), Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_mert(device)
    count = 0
    skipped = 0
    for chart_dir, _dirs, files in charts_dir.walk():
        if "track.mp3" not in files or "maidata.txt" not in files:
            continue
        rel = str(chart_dir.relative_to(charts_dir))
        try:
            _out, created = process_one(chart_dir, rel, cache_dir, device, processor, model)
            count += 1
            if created:
                print(f"[mert-cache] {count}: {rel}")
        except Exception as error:
            skipped += 1
            print(f"[mert-cache] 跳过 {rel}: {error}")
    print(f"[mert-cache] 完成: {count} 首，跳过 {skipped} 首")
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
    print(f"[mert-cache] 自检通过: {features.shape}")


if __name__ == "__main__":
    _self_check()
