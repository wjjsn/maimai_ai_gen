"""ChartDataset — §6 Data Loader for maimai chart generation.

Four-step workflow:
  Step 1: 重建缓存    — walk charts/, compute mel spectrograms, save to .cache/
  Step 2: 验证数据集  — check audio↔chart pairing, parse without errors
  Step 3: 编译谱面    — parse → to_tensor → build index (mel_path + slice coords + tokens)
  Step 4: 迭代读取    — __getitem__ loads mel from disk, slices on the fly
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from mel_cache import main as rebuild_cache
from maidata_parser import compiler


# ── Step 2: Validate dataset ─────────────────────────────────────────────

def validate_dataset(
    charts_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]]]:
    """Check audio↔chart pairing and parse correctness."""
    import hashlib

    charts_dir = Path(charts_dir)
    cache_dir = Path(cache_dir) if cache_dir else charts_dir.parent / ".cache" / "charts"

    def cache_key(rel_path: str) -> str:
        return hashlib.md5(rel_path.encode()).hexdigest()

    valid: list[tuple[Path, Path]] = []
    invalid: list[tuple[Path, str]] = []
    c = compiler()

    for chart_path in sorted(charts_dir.rglob("maidata.txt")):
        chart_dir = chart_path.parent
        track_path = chart_dir / "track.mp3"

        if not track_path.exists():
            invalid.append((chart_path, "missing track.mp3"))
            continue

        rel = str(chart_dir.relative_to(charts_dir))
        mel_path = cache_dir / f"{cache_key(rel)}.npy"
        if not mel_path.exists():
            invalid.append((chart_path, "missing mel cache"))
            continue

        try:
            text = chart_path.read_text(encoding="utf-8")
            c.parse(text)
        except Exception as e:
            invalid.append((chart_path, f"parse error: {e}"))
            continue

        valid.append((chart_path, mel_path))

    return valid, invalid


# ── Step 3: Compile charts & build index ──────────────────────────────────

# Each index entry: (mel_path, start_frame, end_frame, tokens_list)
_IndexEntry = tuple[Path, int, int, list[int]]


def compile_index(
    valid_pairs: list[tuple[Path, Path]],
    level_idx: int,
    sample_rate: int = 22050,
    hop_length: int = 256,
) -> tuple[list[_IndexEntry], int, dict[str, int]]:
    """Build index: for each segment, store mel slice coords + tokens.

    Does NOT load any mel data. Only stores (mel_path, start_frame, end_frame, tokens).
    """
    index: list[_IndexEntry] = []
    max_tokens = 0
    total_segments = 0
    total_frames = 0
    parse_errors = 0
    c = compiler(hop_length=hop_length, sample_rate=sample_rate)
    frames_per_sec = sample_rate / hop_length

    for maidata_path, mel_path in valid_pairs:
        try:
            text = maidata_path.read_text(encoding="utf-8")
            c.parse(text)
        except Exception:
            parse_errors += 1
            continue

        level = c.chart.all_levels[level_idx]
        if level is None:
            continue

        offsets, tensors = c.to_tensor(level_idx=level_idx)
        if not offsets:
            continue

        total_frames += len(level.frames)
        token_lists = [t.tolist() for t in tensors]

        # need mel total frames to compute end boundary for last segment
        mel_arr = np.load(str(mel_path))
        mel_total_frames = mel_arr.shape[1]
        del mel_arr

        for seg_i in range(len(offsets)):
            start_offset = offsets[seg_i]
            if seg_i + 1 < len(offsets):
                end_offset = offsets[seg_i + 1]
            else:
                end_offset = mel_total_frames / frames_per_sec

            start_frame = max(0, int(start_offset * frames_per_sec))
            end_frame = min(mel_total_frames, int(end_offset * frames_per_sec))
            if end_frame <= start_frame:
                end_frame = min(start_frame + 1, mel_total_frames)

            tl = token_lists[seg_i]
            if len(tl) > max_tokens:
                max_tokens = len(tl)

            index.append((mel_path, start_frame, end_frame, tl))
            total_segments += 1

    stats = {
        "charts": len(valid_pairs),
        "segments": total_segments,
        "frames": total_frames,
        "max_tokens": max_tokens,
        "parse_errors": parse_errors,
    }
    return index, max_tokens, stats


# ── Step 4: Dataset — iter read, slice on the fly ─────────────────────────

class ChartDataset(Dataset):
    """Lazy-loading dataset. Slices mel at iteration time.

    Constructor builds index (coords + tokens only).
    __getitem__ loads mel from disk and slices on the fly.
    """

    def __init__(
        self,
        charts_dir: str | Path,
        cache_dir: str | Path | None = None,
        level_idx: int = 4,
        max_tokens: int = 0,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        self.charts_dir = Path(charts_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.charts_dir.parent / ".cache" / "charts"
        self.level_idx = level_idx
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

        # ── Step 1: rebuild cache ──
        print("[dataset] Step 1: rebuilding mel cache...")
        cached, errs = rebuild_cache(charts_dir, self.cache_dir, sample_rate, 1024, hop_length, n_mels)
        print(f"[dataset]   {cached} cached, {errs} errors")

        # ── Step 2: validate ──
        print("[dataset] Step 2: validating dataset...")
        valid, invalid = validate_dataset(charts_dir, self.cache_dir)
        print(f"[dataset]   {len(valid)} valid, {len(invalid)} invalid")
        for path, reason in invalid[:5]:
            print(f"[dataset]   ✗ {path.parent.name}: {reason}")

        # ── Step 3: compile & build index ──
        print("[dataset] Step 3: compiling index...")
        self._index, computed_max, stats = compile_index(
            valid, self.level_idx, sample_rate, hop_length,
        )
        self.max_tokens = max_tokens if max_tokens > 0 else computed_max
        print(f"[dataset]   {stats}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        mel_path, start_frame, end_frame, tl = self._index[idx]

        # ── load mel from disk, slice on the fly ──
        mel_full = np.load(str(mel_path))
        mel_slice = torch.from_numpy(mel_full[:, start_frame:end_frame].copy()).float()
        if mel_slice.shape[1] == 0:
            mel_slice = torch.zeros(self.n_mels, 1)

        # ── pad tokens ──
        length = min(len(tl), self.max_tokens)
        if length >= self.max_tokens:
            tokens = torch.tensor(tl[:self.max_tokens], dtype=torch.int64)
            mask = torch.ones(self.max_tokens, dtype=torch.bool)
        else:
            tokens = torch.tensor(tl + [0] * (self.max_tokens - length), dtype=torch.int64)
            mask = torch.cat([
                torch.ones(length, dtype=torch.bool),
                torch.zeros(self.max_tokens - length, dtype=torch.bool),
            ])

        return {"mel": mel_slice, "tokens": tokens, "mask": mask}


# ── collate ───────────────────────────────────────────────────────────────

def collate_segments(batch: list[dict], mel_frames: int = 0) -> dict:
    """Collate function: pads/trims mel to exactly mel_frames time steps.

    mel_frames defaults to 0 → fall back to max-in-batch (legacy behaviour).
    When mel_frames > 0 each mel is trimmed or zero-padded to that exact width
    so the encoder's positional embedding always matches.
    """
    n_mels = batch[0]["mel"].shape[0]

    if mel_frames <= 0:
        mel_frames = max(item["mel"].shape[1] for item in batch)

    mels = []
    for item in batch:
        mel = item["mel"]                        # (n_mels, T)
        T = mel.shape[1]
        if T < mel_frames:
            pad = torch.zeros(n_mels, mel_frames - T)
            mel = torch.cat([mel, pad], dim=1)
        elif T > mel_frames:
            mel = mel[:, :mel_frames]
        mels.append(mel)

    return {
        "mel": torch.stack(mels),
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


# ── self-check ────────────────────────────────────────────────────────────

def _self_check():
    charts_dir = Path(__file__).resolve().parent.parent / "charts"
    if not charts_dir.exists():
        print("[dataset] charts/ not found, skipping self-check")
        return

    from maidata_parser import SOS, EOS

    ds = ChartDataset(charts_dir, max_tokens=2048, level_idx=4)
    print(f"[dataset] total: {len(ds)} segments")

    if len(ds) > 0:
        item = ds[0]
        t = item["tokens"]
        assert t[0] == SOS, f"Expected SOS, got {t[0]}"
        non_pad = t[t != 0]
        assert non_pad[-1].item() == EOS, f"Expected EOS, got {non_pad[-1].item()}"
        print(f"  mel={item['mel'].shape} tokens={item['tokens'].shape} "
              f"mask_true={item['mask'].sum().item()}")
        print(f"  SOS={t[0].item()} EOS={non_pad[-1].item()} ✓")

    batch = [ds[i] for i in range(min(4, len(ds)))]
    c = collate_segments(batch)
    print(f"  batch: mel={c['mel'].shape} tokens={c['tokens'].shape}")
    print("[dataset] ✓ self-check passed")


if __name__ == "__main__":
    _self_check()
