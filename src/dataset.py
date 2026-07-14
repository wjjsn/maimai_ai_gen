"""训练时现场读取音频，并生成四头事件监督张量。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from audio_features import AudioAugmentation, extract_audio_features, sample_augmentation
from chart import Chart, Frame, HoldData, Level, Note, NoteType, TapType
from config import CONFIG
from maidata_parser import parse_maidata
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, TRACK_COUNT, _slide_branches


@dataclass(frozen=True)
class SongEntry:
    chart_path: Path
    audio_path: Path
    level_query: float | None


def _target_scale() -> np.ndarray:
    return np.array((8, 2, round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec), round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)), dtype=np.float32)


def _note_end_sec(frame: Frame, note: Note) -> float:
    if note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
        return frame.time_sec + note.data.holdTime
    if note.type is NoteType.SLIDE:
        return frame.time_sec + max((sum(
            segment.wait_duration + segment.trace_duration for segment in branch
        ) for branch in _slide_branches(note)), default=0.0)
    return frame.time_sec


def validate_chart_audio_alignment(chart: Chart, audio_duration_sec: float, level_idx: int) -> None:
    level = chart.all_levels[level_idx]
    if level is None:
        raise ValueError("缺少目标难度")
    tolerance = 1 / CONFIG.audio.frames_per_sec
    for frame in level.frames:
        start = frame.time_sec + chart.first_sec
        if start < 0:
            raise ValueError(f"{start:.3f}s 音符早于音频开头，跳过整首歌")
        if start >= audio_duration_sec:
            raise ValueError(f"{start:.3f}s 音符晚于音频结尾 {audio_duration_sec:.3f}s，跳过整首歌")
        for note in frame.notes:
            end = _note_end_sec(frame, note) + chart.first_sec
            if end > audio_duration_sec + tolerance:
                raise ValueError(f"{end:.3f}s 持续音晚于音频结尾 {audio_duration_sec:.3f}s，跳过整首歌")


def chart_to_targets(chart: Chart, length: int, level_idx: int, augmentation: AudioAugmentation | None = None) -> np.ndarray:
    level = chart.all_levels[level_idx]
    if level is None:
        raise ValueError("缺少目标难度")
    rate = CONFIG.audio.frames_per_sec
    speed = augmentation.speed if augmentation is not None else 1.0
    shift = augmentation.shift_sec if augmentation is not None else 0.0
    targets = np.zeros((length, TRACK_COUNT), dtype=np.float32)
    for frame in level.frames:
        start = round(((frame.time_sec + chart.first_sec) / speed + shift) * rate)
        if not 0 <= start < length:
            continue
        for note in frame.notes:
            if note.type in (NoteType.TAP, NoteType.TOUCH):
                targets[start, TAP_COUNT] += 1
            elif note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                duration = max(1, round(note.data.holdTime / speed * rate))
                slot = int(targets[start, HOLD_START_COUNT])
                if slot == 2:
                    raise ValueError(f"{start / rate:.3f}s 同时开始两个以上长按，跳过整首歌")
                targets[start, HOLD_START_COUNT] += 1
                targets[start, HOLD_DURATION_1 + slot] = duration
            elif note.type is NoteType.SLIDE:
                for branch in _slide_branches(note):
                    if not all(segment.is_default_wait for segment in branch):
                        raise ValueError(f"{start / rate:.3f}s SLIDE 使用非默认等待，跳过整首歌")
                    duration = max(1, round(sum(
                        segment.wait_duration + segment.trace_duration for segment in branch
                    ) / speed * rate))
                    slot = int(targets[start, HOLD_START_COUNT])
                    if slot == 2:
                        raise ValueError(f"{start / rate:.3f}s 同时开始两个以上长按，跳过整首歌")
                    targets[start, HOLD_START_COUNT] += 1
                    targets[start, HOLD_DURATION_1 + slot] = duration
    scale = _target_scale()
    if (targets > scale).any():
        raise ValueError("事件数量或长按时长超出归一化范围，跳过整首歌")
    return targets / scale


def discover_songs(charts_dir: Path, level_idx: int) -> tuple[list[SongEntry], list[str]]:
    entries, skipped = [], []
    chart_paths = sorted(charts_dir.rglob("maidata.txt"))
    for chart_path in tqdm(chart_paths, desc="扫描训练歌曲", unit="首"):
        audio_path = chart_path.parent / "track.mp3"
        if not audio_path.is_file():
            continue
        relative = str(chart_path.parent.relative_to(charts_dir))
        try:
            chart = parse_maidata(chart_path.read_text(encoding="utf-8"))
            audio_duration_sec = AudioDecoder(audio_path).metadata.duration_seconds
            if audio_duration_sec is None or audio_duration_sec <= 0:
                raise ValueError("无法读取有效音频时长")
            validate_chart_audio_alignment(chart, audio_duration_sec, level_idx)
            length = max(1, round(audio_duration_sec * CONFIG.audio.frames_per_sec) + 1)
            chart_to_targets(chart, length, level_idx)
            entries.append(SongEntry(chart_path, audio_path, chart.all_levels[level_idx].level_query))
        except Exception as error:
            skipped.append(f"{relative}: {error}")
    return entries, skipped


class SongDataset(Dataset):
    def __init__(self, entries: list[SongEntry], level_idx: int, augment: bool = False):
        self.entries, self.level_idx, self.augment = entries, level_idx, augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        augmentation = sample_augmentation() if self.augment else None
        features = extract_audio_features(entry.audio_path, augmentation=augmentation)
        chart = parse_maidata(entry.chart_path.read_text(encoding="utf-8"))
        return {"features": torch.from_numpy(features), "events": torch.from_numpy(chart_to_targets(chart, len(features), self.level_idx, augmentation)), "entry": entry}


def collate_songs(items: list[dict]) -> dict:
    max_length = max(item["features"].shape[0] for item in items)
    batch, feature_dim = len(items), items[0]["features"].shape[1]
    features = torch.zeros(batch, max_length, feature_dim)
    events = torch.zeros(batch, max_length, TRACK_COUNT)
    mask = torch.zeros(batch, max_length, dtype=torch.bool)
    for i, item in enumerate(items):
        length = item["features"].shape[0]
        features[i, :length], events[i, :length], mask[i, :length] = item["features"], item["events"], True
    return {"features": features, "events": events, "mask": mask, "entries": [item["entry"] for item in items]}


def _self_check() -> None:
    chart = Chart(first_sec=0.2)
    chart.all_levels[5] = Level("master", 14, [Frame((Note(NoteType.TAP, TapType.LANE1),), 0.1)])
    targets = chart_to_targets(chart, 100, 5)
    assert targets[60, TAP_COUNT] == 1 / _target_scale()[TAP_COUNT]

    augmented = chart_to_targets(chart, 100, 5, AudioAugmentation(speed=2.0, shift_sec=0.1))
    assert augmented[50, TAP_COUNT] == 1 / _target_scale()[TAP_COUNT]

    long_chart = Chart(first_sec=0.2)
    long_chart.all_levels[5] = Level("master", 14, [Frame((
        Note(NoteType.HOLD, HoldData(TapType.LANE1, 0.2)),
    ), 0.7)])
    try:
        validate_chart_audio_alignment(long_chart, 1.0, 5)
    except ValueError as error:
        assert "持续音晚于音频结尾" in str(error)
    else:
        raise AssertionError("超出音频的持续音必须被拒绝")
    print("[dataset] 自检通过")


if __name__ == "__main__":
    _self_check()
