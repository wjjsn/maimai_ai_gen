"""Walk charts/, find dirs with track.mp3 + maidata.txt, compute mel spectrogram, save to .cache/."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torchaudio

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "charts"

transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)

    
def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def process_one(chart_dir: Path, rel_path: str) -> Path:
    out = CACHE_DIR / f"{cache_key(rel_path)}.npy"
    if out.exists():
        return out
    waveform, sr = torchaudio.load(str(chart_dir / "track.mp3"))
    if sr != 22050:
        waveform = torchaudio.functional.resample(waveform, sr, 22050)
    mel = transform(waveform)
    log_mel = torch.log(mel + 1e-6).numpy()
    print(f"    ┌形状: {log_mel.shape}")
    np.save(out, log_mel)
    return out


def main():
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    found = 0
    skipped = 0
    for chart_dir, _dirs, files in CHARTS_DIR.walk():
        if "track.mp3" not in files or "maidata.txt" not in files:
            continue
        rel = chart_dir.relative_to(CHARTS_DIR)
        try:
            out = process_one(chart_dir, str(rel))
            found += 1
            print(f"[{found}] {rel} -> {out.name}")
        except Exception as e:
            skipped += 1
            print(f"[ERR] {rel}: {e}")
    print(f"\nDone: {found} cached, {skipped} errors")


if __name__ == "__main__":
    main()
