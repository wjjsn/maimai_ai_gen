"""Walk charts/, find dirs with track.mp3 + maidata.txt, compute mel spectrogram, save to .cache/."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from config import CONFIG

CHARTS_DIR = CONFIG.paths.charts_dir
CACHE_DIR = CONFIG.paths.mel_cache_dir



    
def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def _metadata_path(out: Path) -> Path:
    return out.with_suffix(".json")


def _file_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_is_current(out: Path, track_path: Path, config: dict[str, int]) -> bool:
    metadata_path = _metadata_path(out)
    if not out.exists():
        return False
    if not metadata_path.exists():
        # 无法从数组形状推断采样率、FFT 和 hop，不能把未知来源的旧缓存认证为当前配置。
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata["track"] == _file_state(track_path) and metadata["config"] == config
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def process_one(
    chart_dir: Path,
    rel_path: str,
    cache_dir: Path = CACHE_DIR,
    sample_rate=CONFIG.audio.sample_rate,
    n_fft=CONFIG.audio.n_fft,
    hop_length=CONFIG.audio.hop_length,
    n_mels=CONFIG.audio.n_mels,
) -> tuple[Path, bool]:
    out = cache_dir / f"{cache_key(rel_path)}.npy"
    track_path = chart_dir / "track.mp3"
    config = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    if _cache_is_current(out, track_path, config):
        return out, False
    waveform, sr = torchaudio.load(str(track_path))
    if waveform.shape[0] > 1:
         waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
         waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    )
    mel = transform(waveform)
    log_mel = torch.log(mel + 1e-6).squeeze(0).numpy()
    print(f"    ┌形状: {log_mel.shape}=(n_mels`梅尔频带数`, T`时间`)")
    temp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with temp.open("wb") as file:
        np.save(file, log_mel)
    temp.replace(out)
    _write_json(_metadata_path(out), {
        "cache_version": 1,
        "config": config,
        "mel_shape": list(log_mel.shape),
        "track": _file_state(track_path),
    })
    return out, True


def main(
    charts_dir=CHARTS_DIR,
    cache_dir=CACHE_DIR,
    sample_rate=CONFIG.audio.sample_rate,
    n_fft=CONFIG.audio.n_fft,
    hop_length=CONFIG.audio.hop_length,
    n_mels=CONFIG.audio.n_mels,
):
    cache_dir.mkdir(exist_ok=True, parents=True)
    found = 0
    skipped = 0
    charts_dir = Path(charts_dir)
    cache_dir = Path(cache_dir)
    for chart_dir, _dirs, files in charts_dir.walk():
        if "track.mp3" not in files or "maidata.txt" not in files:
            continue
        rel = chart_dir.relative_to(charts_dir)
        try:
            out, created = process_one(chart_dir, str(rel), cache_dir, sample_rate, n_fft, hop_length, n_mels)
            found += 1
            if created:
                print(f"[{found}] {rel} -> {out.name}")
        except Exception as e:
            skipped += 1
            print(f"[ERR] {rel}: {e}")
    print(f"\nDone: {found} cached, {skipped} errors")
    return found, skipped


if __name__ == "__main__":
    main()
