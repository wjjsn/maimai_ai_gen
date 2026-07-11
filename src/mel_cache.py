"""Walk charts/, find dirs with track.mp3 + maidata.txt, compute mel spectrogram, save to .cache/."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torchaudio

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "charts"



    
def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def process_one(chart_dir: Path, rel_path: str, cache_dir: Path = CACHE_DIR, sample_rate = 22050, n_fft=1024, hop_length = 256, n_mels = 80) -> tuple[Path, bool]:
    out = cache_dir / f"{cache_key(rel_path)}.npy"
    if out.exists():
        return out, False
    waveform, sr = torchaudio.load(str(chart_dir / "track.mp3"))
    if waveform.shape[0] > 1:
         waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 22050:
        waveform = torchaudio.functional.resample(waveform, sr, 22050)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    )
    mel = transform(waveform)
    log_mel = torch.log(mel + 1e-6).squeeze(0).numpy()
    print(f"    ┌形状: {log_mel.shape}=(n_mels`梅尔频带数`, T`时间`)")
    np.save(out, log_mel)
    return out, True


def main(charts_dir = CHARTS_DIR, cache_dir = CACHE_DIR, sample_rate = 22050, n_fft=1024, hop_length = 256, n_mels = 80):
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
