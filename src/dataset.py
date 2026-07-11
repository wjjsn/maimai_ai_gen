"""ChartDataset — §6 Data Loader for maimai chart generation.

Four-step workflow:
  Step 1: 重建缓存    — walk charts/, compute mel spectrograms, save to .cache/
  Step 2: 验证数据集  — check audio↔chart pairing, parse without errors
  Step 3: 编译谱面    — parse → to_tensor → build index (mel_path + slice coords + tokens)
  Step 4: 迭代读取    — __getitem__ loads mel from disk, slices on the fly
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset
from chart_cache import ensure_chart_cache
from maidata_parser import EOS, FRAME_END, FRAME_START, LANE_BASE, SOS, TOUCH_BASE, compiler


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

PREFIX_START_SEC = 6.0
TARGET_START_SEC = 12.0
TARGET_END_SEC = 22.0


class _IndexEntry(NamedTuple):
    mel_path: Path
    source_start_frame: int
    source_end_frame: int
    left_pad_frames: int
    window_start_sec: float
    target_start_sec: float
    tokens: list[int]
    loss_mask: list[bool]


@dataclass
class _CachedIndexEntry:
    mel_path: Path
    source_start_frame: int
    source_end_frame: int
    left_pad_frames: int
    window_start_sec: float
    target_start_sec: float
    tokens: np.ndarray
    loss_start: int

    @property
    def loss_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.tokens), dtype=bool)
        mask[self.loss_start:] = True
        return mask


class _CachedIndex:
    """只保留窗口行号，具体 token 与元数据按需从 mmap 读取。"""

    def __init__(self, cache_path: Path, charts_dir: Path, cache_dir: Path, selected_charts: set[Path] | None = None):
        import json

        manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
        self.rows = np.load(cache_path / "entries.npy", mmap_mode="r")
        self.tokens = np.load(cache_path / "tokens.npy", mmap_mode="r")
        self.mel_paths = [cache_dir.parent / chart["mel"] for chart in manifest["charts"]]
        self.chart_paths = [charts_dir / chart["chart"] for chart in manifest["charts"]]
        if selected_charts is None:
            self.indices = np.arange(len(self.rows), dtype=np.int64)
        else:
            selected_ids = {i for i, path in enumerate(self.chart_paths) if path in selected_charts}
            self.indices = np.flatnonzero(np.isin(self.rows["chart_id"], list(selected_ids)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> _CachedIndexEntry:
        row = self.rows[self.indices[index]]
        token_start = int(row["token_start"])
        token_end = token_start + int(row["token_length"])
        return _CachedIndexEntry(
            mel_path=self.mel_paths[int(row["chart_id"])],
            source_start_frame=int(row["source_start_frame"]),
            source_end_frame=int(row["source_end_frame"]),
            left_pad_frames=int(row["left_pad_frames"]),
            window_start_sec=float(row["window_start_sec"]),
            target_start_sec=float(row["target_start_sec"]),
            tokens=self.tokens[token_start:token_end],
            loss_start=int(row["loss_start"]),
        )


def rotate_token_id(token: int, steps: int) -> int:
    """把 8 轨相关 token 向右旋转 steps 次；旋转 8 次必须还原。"""
    steps %= 8
    if steps == 0:
        return token

    if LANE_BASE <= token < LANE_BASE + 8:
        return LANE_BASE + ((token - LANE_BASE + steps) % 8)

    if TOUCH_BASE <= token < TOUCH_BASE + 33:
        offset = token - TOUCH_BASE
        if 0 <= offset < 8:       # A1~A8
            return TOUCH_BASE + ((offset + steps) % 8)
        if 8 <= offset < 16:      # B1~B8
            return TOUCH_BASE + 8 + ((offset - 8 + steps) % 8)
        if offset == 16:          # C 不随 8 轨旋转
            return token
        if 17 <= offset < 25:     # D1~D8
            return TOUCH_BASE + 17 + ((offset - 17 + steps) % 8)
        if 25 <= offset < 33:     # E1~E8
            return TOUCH_BASE + 25 + ((offset - 25 + steps) % 8)

    return token


def rotate_token_list(tokens: list[int], steps: int) -> list[int]:
    return [rotate_token_id(token, steps) for token in tokens]


class RotatedDataset(Dataset):
    """把基础数据集扩成 8 个方向；只旋转 token，不改音频。"""

    def __init__(self, base: Dataset, rotations: int = 8):
        self.base = base
        self.rotations = rotations

    def __len__(self) -> int:
        return len(self.base) * self.rotations

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx // self.rotations]
        steps = idx % self.rotations
        if steps == 0:
            return item
        rotated = dict(item)
        rotated["tokens"] = torch.tensor(
            rotate_token_list(item["tokens"].tolist(), steps),
            dtype=item["tokens"].dtype,
        )
        return rotated


def compile_index(
    valid_pairs: list[tuple[Path, Path]],
    level_idx: int,
    sample_rate: int = 22050,
    hop_length: int = 256,
    stride_sec: float = 1.0,
    mel_frames: int = 3000,
) -> tuple[list[_IndexEntry], int, dict[str, int]]:
    """构建平移窗口：6..12 秒谱面前缀，12..22 秒训练目标。"""
    index: list[_IndexEntry] = []
    max_tokens = 0
    total_windows = 0
    total_frames = 0
    parse_errors = 0
    c = compiler(hop_length=hop_length, sample_rate=sample_rate)
    frames_per_sec = sample_rate / hop_length
    window_sec = mel_frames / frames_per_sec

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

        total_frames += len(level.frames)

        # 只读 shape；实际 mel 数据仍在 __getitem__ 里按需 mmap。
        mel_arr = np.load(str(mel_path))
        mel_total_frames = mel_arr.shape[1]
        del mel_arr
        song_end_sec = mel_total_frames / frames_per_sec

        target_start = 0.0
        while target_start < song_end_sec:
            window_start = target_start - TARGET_START_SEC
            logical_start_frame = round(window_start * frames_per_sec)
            source_start = max(0, logical_start_frame)
            source_end = min(mel_total_frames, logical_start_frame + mel_frames)
            left_pad = source_start - logical_start_frame

            tl = [SOS]
            loss_mask = [False]
            for frame in level.frames:
                if not frame.notes:
                    continue
                rel_cs = round((frame.time_sec - window_start) * 100)
                is_prefix = round(PREFIX_START_SEC * 100) <= rel_cs < round(TARGET_START_SEC * 100)
                is_target = round(TARGET_START_SEC * 100) <= rel_cs < round(TARGET_END_SEC * 100)
                if not (is_prefix or is_target):
                    continue

                frame_tokens = [FRAME_START, c._ts_token(rel_cs / 100.0)]
                for note in frame.notes:
                    frame_tokens.extend(c._encode_note_tokens(note))
                frame_tokens.append(FRAME_END)
                tl.extend(frame_tokens)
                loss_mask.extend([is_target] * len(frame_tokens))
            tl.append(EOS)
            loss_mask.append(True)

            if len(tl) > max_tokens:
                max_tokens = len(tl)

            index.append(_IndexEntry(
                mel_path=mel_path,
                source_start_frame=source_start,
                source_end_frame=max(source_start, source_end),
                left_pad_frames=left_pad,
                window_start_sec=window_start,
                target_start_sec=target_start,
                tokens=tl,
                loss_mask=loss_mask,
            ))
            total_windows += 1
            target_start += stride_sec

    stats = {
        "charts": len(valid_pairs),
        "windows": total_windows,
        "frames": total_frames,
        "max_tokens": max_tokens,
        "parse_errors": parse_errors,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "prefix_start_sec": PREFIX_START_SEC,
        "target_start_sec": TARGET_START_SEC,
        "target_end_sec": TARGET_END_SEC,
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
        level_idx: int = 5,
        max_tokens: int = 0,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
        stride_sec: float = 1.0,
        mel_frames: int = 3000,
        valid_pairs: list[tuple[Path, Path]] | None = None,
        chart_limit: int = 0,
        seed: int = 42,
    ):
        self.charts_dir = Path(charts_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.charts_dir.parent / ".cache" / "charts"
        self.level_idx = level_idx
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.stride_sec = stride_sec
        self.mel_frames = mel_frames
        self._mel_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        cache_path = ensure_chart_cache(
            self.charts_dir,
            self.cache_dir,
            level_idx=level_idx,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels,
            stride_sec=stride_sec,
            mel_frames=mel_frames,
            build_mel=valid_pairs is None,
        )
        all_index = _CachedIndex(cache_path, self.charts_dir, self.cache_dir)
        all_valid = list(zip(all_index.chart_paths, all_index.mel_paths))
        valid = list(valid_pairs) if valid_pairs is not None else all_valid
        if chart_limit > 0 and len(valid) > chart_limit:
            rng = random.Random(seed)
            valid = rng.sample(valid, chart_limit)
            valid.sort()
        self.valid_pairs = valid
        self._index = _CachedIndex(cache_path, self.charts_dir, self.cache_dir, {path for path, _ in valid})
        computed_max = max((len(entry.tokens) for entry in self._index), default=0)
        self.max_tokens = max_tokens if max_tokens > 0 else computed_max
        print(f"[dataset] 缓存索引: {len(valid)} 首歌，{len(self._index)} 个窗口，max_tokens={self.max_tokens}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        entry = self._index[idx]

        # 每个 DataLoader worker 复用 mmap，避免同一首歌的分段反复从磁盘读整文件。
        mel_full = self._mel_cache.get(entry.mel_path)
        if mel_full is None:
            mel_full = np.load(str(entry.mel_path), mmap_mode="r")
            self._mel_cache[entry.mel_path] = mel_full
            # ponytail: 每个 worker 最多保持 8 个 mmap；需要更多时再按命中率调大。
            if len(self._mel_cache) > 8:
                self._mel_cache.popitem(last=False)
        else:
            self._mel_cache.move_to_end(entry.mel_path)
        source = torch.from_numpy(
            mel_full[:, entry.source_start_frame:entry.source_end_frame].copy()
        ).float()
        mel_slice = torch.zeros(self.n_mels, self.mel_frames)
        copy_len = min(source.shape[1], self.mel_frames - entry.left_pad_frames)
        if copy_len > 0:
            mel_slice[:, entry.left_pad_frames:entry.left_pad_frames + copy_len] = source[:, :copy_len]

        # ── pad tokens ──
        tl = entry.tokens
        if len(tl) > self.max_tokens:
            raise ValueError(f"窗口 token 数 {len(tl)} 超过 max_tokens={self.max_tokens}，拒绝截断合法序列")
        length = len(tl)
        if length >= self.max_tokens:
            tokens = torch.from_numpy(np.asarray(tl[:self.max_tokens], dtype=np.int64))
            mask = torch.ones(self.max_tokens, dtype=torch.bool)
        else:
            tokens = torch.zeros(self.max_tokens, dtype=torch.int64)
            tokens[:length] = torch.from_numpy(np.asarray(tl, dtype=np.int64))
            mask = torch.cat([
                torch.ones(length, dtype=torch.bool),
                torch.zeros(self.max_tokens - length, dtype=torch.bool),
            ])

        loss_mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        loss_mask[entry.loss_start:length] = True
        return {
            "mel": mel_slice,
            "tokens": tokens,
            "mask": mask,
            "loss_mask": loss_mask,
            "window_start_sec": entry.window_start_sec,
            "target_start_sec": entry.target_start_sec,
        }


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

    mels = torch.zeros(len(batch), n_mels, mel_frames)
    for i, item in enumerate(batch):
        mel = item["mel"]                        # (n_mels, T)
        T = mel.shape[1]
        mels[i, :, :min(T, mel_frames)] = mel[:, :mel_frames]

    return {
        "mel": mels,
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "window_start_sec": torch.tensor([item["window_start_sec"] for item in batch]),
        "target_start_sec": torch.tensor([item["target_start_sec"] for item in batch]),
    }


# ── self-check ────────────────────────────────────────────────────────────

def _self_check():
    charts_dir = Path(__file__).resolve().parent.parent / "charts"
    if not charts_dir.exists():
        print("[dataset] charts/ not found, skipping self-check")
        return

    from maidata_parser import SOS, EOS

    ds = ChartDataset(charts_dir, max_tokens=2048, level_idx=5)
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
