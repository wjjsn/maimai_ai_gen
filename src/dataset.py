"""单曲多难度滑窗数据集，以及谱面四列张量往返转换。"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
from urllib.request import urlopen

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from chart import Chart, Frame, HoldData, Level, Note, NoteType, SlideSegment, TapType
from config import CONFIG, ROOT_DIR
from maidata_parser import generate_maidata, parse_maidata


TAP_COUNT = 0
HOLD_START_COUNT = 1
HOLD_DURATION_1 = 2
HOLD_DURATION_2 = 3
TRACK_COUNT = 4
WINDOW_FRAMES = 1024
TRAIN_STRIDE = 10
MUSIC_DATA_URL = "https://www.diving-fish.com/api/maimaidxprober/music_data"
MUSIC_DATA_CACHE = ROOT_DIR / ".cache" / "diving-fish-music-data.json"


@dataclass(frozen=True)
class SongLevel:
    level_idx: int
    difficulty: float
    tracks: np.ndarray


@dataclass(frozen=True)
class SongData:
    chart_path: Path
    audio_path: Path
    title: str
    music_id: str
    features: np.ndarray
    levels: tuple[SongLevel, ...]
    music_data_digest: str


def _slide_branches(note: Note) -> list[list[SlideSegment]]:
    branches: list[list[SlideSegment]] = []
    for segment in note.data:
        if not branches or segment.start_lane != branches[-1][-1].end_lane:
            branches.append([])
        branches[-1].append(segment)
    return branches


def chart_to_tracks(chart: Chart, level_idx: int, length: int | None = None) -> np.ndarray:
    level = chart.all_levels[level_idx]
    if level is None:
        raise ValueError("缺少目标难度")
    rate = CONFIG.audio.frames_per_sec
    tap_events: list[int] = []
    hold_events: list[tuple[int, int]] = []
    required_length = 1
    for frame in level.frames:
        start = round((frame.time_sec + chart.first_sec) * rate)
        if start < 0:
            raise ValueError("音符时间不能为负")
        required_length = max(required_length, start + 1)
        for note in frame.notes:
            if note.type in (NoteType.TAP, NoteType.TOUCH):
                tap_events.append(start)
            elif note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                duration = max(1, round(note.data.holdTime * rate))
                hold_events.append((start, duration))
                required_length = max(required_length, start + duration)
            elif note.type is NoteType.SLIDE:
                for branch in _slide_branches(note):
                    if not all(segment.is_default_wait for segment in branch):
                        raise ValueError(f"{start / rate:.3f}s SLIDE 使用非默认等待，跳过整首歌")
                    duration = max(1, round(sum(
                        segment.wait_duration + segment.trace_duration for segment in branch
                    ) * rate))
                    hold_events.append((start, duration))
                    required_length = max(required_length, start + duration)
    if length is None:
        length = required_length
    if length < required_length:
        raise ValueError(f"张量长度不足，需要至少 {required_length} 帧")
    tracks = np.zeros((length, TRACK_COUNT), dtype=np.int32)
    for start in tap_events:
        tracks[start, TAP_COUNT] += 1
    for start, duration in hold_events:
        slot = int(tracks[start, HOLD_START_COUNT])
        if slot == 2:
            raise ValueError(f"{start / rate:.3f}s 同时开始两个以上长按，跳过整首歌")
        tracks[start, HOLD_START_COUNT] += 1
        tracks[start, HOLD_DURATION_1 + slot] = duration
    return tracks


def maidata_to_tracks(text: str, level_idx: int = 5, length: int | None = None) -> np.ndarray:
    return chart_to_tracks(parse_maidata(text), level_idx, length)


def tracks_to_chart(
    tracks: np.ndarray, level_idx: int = 5, title: str = "generated", difficulty: float = 0.0,
) -> Chart:
    tracks = np.asarray(tracks)
    if tracks.ndim != 2 or tracks.shape[1] != TRACK_COUNT:
        raise ValueError(f"张量形状必须为 (T, {TRACK_COUNT})")
    if (tracks < 0).any() or not np.equal(tracks, np.floor(tracks)).all():
        raise ValueError("张量必须包含非负整数")
    if (tracks[:, HOLD_START_COUNT] > 2).any():
        raise ValueError("Hold Start Count 不能超过 2")
    rate = CONFIG.audio.frames_per_sec
    grouped: dict[int, list[Note]] = {}
    for frame, values in enumerate(tracks.astype(np.int64, copy=False)):
        for index in range(values[TAP_COUNT]):
            grouped.setdefault(frame, []).append(Note(
                NoteType.TAP, (TapType.LANE1, TapType.LANE8)[index % 2],
            ))
        for slot in range(values[HOLD_START_COUNT]):
            duration = values[HOLD_DURATION_1 + slot]
            if duration:
                grouped.setdefault(frame, []).append(Note(
                    NoteType.HOLD,
                    HoldData((TapType.LANE2, TapType.LANE3)[slot], duration / rate),
                ))
    chart = Chart(title=title, artist="generated")
    chart.all_levels[level_idx] = Level(f"level_{level_idx + 1}", difficulty, [
        Frame(tuple(notes), frame / rate) for frame, notes in sorted(grouped.items())
    ])
    return chart


def tracks_to_maidata(
    tracks: np.ndarray, level_idx: int = 5, title: str = "generated", difficulty: float = 0.0,
) -> str:
    return generate_maidata(tracks_to_chart(tracks, level_idx, title, difficulty))


def txt2tensor2txt(text: str, level_idx: int = 5) -> tuple[np.ndarray, str]:
    tracks = maidata_to_tracks(text, level_idx)
    return tracks, tracks_to_maidata(tracks, level_idx)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        return torchaudio.load(str(path))
    except RuntimeError as original:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
             "-ar", str(CONFIG.audio.sample_rate), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
        if not samples.size:
            message = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"音频解码失败: {path}: {message}") from original
        return torch.from_numpy(samples).unsqueeze(0), CONFIG.audio.sample_rate


def extract_log_mel(path: str | Path) -> np.ndarray:
    waveform, sample_rate = load_audio(path)
    waveform = waveform.mean(dim=0, keepdim=True).float()
    if sample_rate != CONFIG.audio.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, CONFIG.audio.sample_rate)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG.audio.sample_rate,
        n_fft=CONFIG.audio.n_fft,
        win_length=CONFIG.audio.win_length,
        hop_length=CONFIG.audio.hop_length,
        f_min=CONFIG.audio.f_min,
        f_max=CONFIG.audio.f_max,
        n_mels=CONFIG.audio.n_mels,
        power=2.0,
    )(waveform)
    return torch.log(mel.clamp_min(1e-10)).squeeze(0).transpose(0, 1).contiguous().numpy()


def _validate_music_data(data: object) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("歌曲数据必须是非空数组")
    for index, song in enumerate(data):
        if not isinstance(song, dict) or not all(key in song for key in ("id", "title", "type", "ds")):
            raise ValueError(f"第 {index} 首歌曲缺少必要字段")
        if not isinstance(song["title"], str) or song["type"] not in ("SD", "DX") or not isinstance(song["ds"], list):
            raise ValueError(f"第 {index} 首歌曲字段类型错误")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in song["ds"]):
            raise ValueError(f"第 {index} 首歌曲含无效定数")
    return data


def _music_data_digest(data: list[dict]) -> str:
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def load_music_data() -> tuple[list[dict], str]:
    if MUSIC_DATA_CACHE.is_file():
        try:
            data = _validate_music_data(json.loads(MUSIC_DATA_CACHE.read_text(encoding="utf-8")))
            return data, _music_data_digest(data)
        except Exception as error:
            raise RuntimeError(f"Diving-Fish 本地缓存损坏，请删除后重新下载: {error}") from error
    try:
        with urlopen(MUSIC_DATA_URL, timeout=30) as response:
            data = _validate_music_data(json.load(response))
        MUSIC_DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = MUSIC_DATA_CACHE.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, MUSIC_DATA_CACHE)
        return data, _music_data_digest(data)
    except Exception as error:
        raise RuntimeError(f"无法获取 Diving-Fish 歌曲数据: {error}") from error


def _normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title).strip()
    title = re.sub(r"\s*\[(?:DX|ST)\]\s*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title)


def match_music(text: str, chart_path: Path, music_data: list[dict]) -> dict | None:
    match = re.search(r"^&title=([^\r\n]+)", text, re.MULTILINE)
    title = match.group(1).strip() if match else chart_path.parent.name
    candidates = [song for song in music_data if _normalize_title(song["title"]) == _normalize_title(title)]
    suffix = re.search(r"\[(DX|ST)\]\s*$", title, re.IGNORECASE)
    if suffix:
        expected = suffix.group(1).upper().replace("ST", "SD")
        candidates = [song for song in candidates if song["type"] == expected]
    shortid = re.search(r"^&shortid=(\d+)", text, re.MULTILINE)
    if shortid:
        matched = [song for song in candidates if str(song["id"]) == shortid.group(1)]
        return matched[0] if len(matched) == 1 else None
    folder_id = re.match(r"(\d+)_", chart_path.parent.name)
    if folder_id:
        matched = [song for song in candidates if str(song["id"]) == folder_id.group(1)]
        if len(matched) == 1:
            return matched[0]
    if not suffix:
        standard = [song for song in candidates if song["type"] == "SD"]
        if len(standard) == 1:
            return standard[0]
    return candidates[0] if len(candidates) == 1 else None


def load_overfit_song(charts_dir: Path = CONFIG.paths.charts_dir) -> SongData:
    music_data, digest = load_music_data()
    errors: list[str] = []
    for chart_path in sorted(charts_dir.rglob("maidata.txt")):
        audio_path = chart_path.parent / "track.mp3"
        if not audio_path.is_file():
            continue
        relative = chart_path.parent.relative_to(charts_dir)
        try:
            text = chart_path.read_text(encoding="utf-8")
            song = match_music(text, chart_path, music_data)
            if song is None:
                raise ValueError("无法匹配 Diving-Fish 歌曲数据")
            chart = parse_maidata(text)
            features = extract_log_mel(audio_path)
            levels: list[SongLevel] = []
            for ds_index, difficulty in enumerate(song["ds"]):
                level_idx = ds_index + 2
                level = chart.all_levels[level_idx]
                if level is None or not level.frames:
                    continue
                tracks = chart_to_tracks(chart, level_idx, len(features))
                maximum_duration = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
                if tracks[:, TAP_COUNT].max(initial=0) > 8:
                    raise ValueError(f"难度 {level_idx} 同一帧短音超过 8 个")
                if tracks[:, HOLD_DURATION_1:].max(initial=0) > maximum_duration:
                    raise ValueError(f"难度 {level_idx} 持续音超过配置上限")
                levels.append(SongLevel(level_idx, float(difficulty), tracks))
            if not levels:
                raise ValueError("没有可用的 Basic 至 Re:Master 谱面")
            return SongData(
                chart_path, audio_path, str(song["title"]), str(song["id"]),
                features, tuple(levels), digest,
            )
        except Exception as error:
            errors.append(f"{relative}: {error}")
    details = "\n".join(errors[:20])
    raise ValueError(f"没有可用于过拟合的歌曲\n{details}")


class OverfitWindowDataset(Dataset):
    def __init__(self, song: SongData, window_frames: int = WINDOW_FRAMES, stride: int = TRAIN_STRIDE):
        if window_frames <= 0 or stride <= 0:
            raise ValueError("窗口长度和步长必须大于 0")
        self.song = song
        self.window_frames = window_frames
        self.index = [
            (level_index, start)
            for level_index in range(len(song.levels))
            for start in range(0, len(song.features), stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        level_index, start = self.index[index]
        level = self.song.levels[level_index]
        end = min(start + self.window_frames, len(self.song.features))
        length = end - start
        features = np.zeros((self.window_frames, self.song.features.shape[1]), dtype=np.float32)
        events = np.zeros((self.window_frames, TRACK_COUNT), dtype=np.int64)
        features[:length] = self.song.features[start:end]
        events[:length] = level.tracks[start:end]
        mask = np.zeros(self.window_frames, dtype=bool)
        mask[:length] = True
        return {
            "features": torch.from_numpy(features),
            "events": torch.from_numpy(events),
            "mask": torch.from_numpy(mask),
            "difficulty": torch.tensor(level.difficulty, dtype=torch.float32),
            "level_idx": torch.tensor(level.level_idx, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def validate_all_songs() -> None:
    paths = sorted(CONFIG.paths.charts_dir.rglob("maidata.txt"))
    if not paths:
        print(f"[dataset] 谱面目录为空，跳过全曲往返验证: {CONFIG.paths.charts_dir}")
        return
    output_dir = ROOT_DIR / "tmp" / "txt2tensor2txt"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    passed, failed = 0, []
    for path in paths:
        relative = path.relative_to(CONFIG.paths.charts_dir)
        try:
            source = path.read_text(encoding="utf-8")
            tracks, generated = txt2tensor2txt(source, CONFIG.training.level_idx)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(generated, encoding="utf-8")
            restored = maidata_to_tracks(generated, CONFIG.training.level_idx, len(tracks))
            if not np.array_equal(tracks, restored):
                raise ValueError("txt -> tensor -> txt -> tensor 不一致")
            passed += 1
        except Exception as error:
            failed.append(f"{relative.parent}: {error}")
    print(f"[dataset] 全曲往返验证: 总数={len(paths)} 通过={passed} 跳过={len(failed)}")
    for error in failed[:20]:
        print(f"  {error}")


def _self_check() -> None:
    chart = Chart(title="test", first_sec=0.05)
    chart.all_levels[5] = Level("master", 14.0, [
        Frame((Note(NoteType.TAP, TapType.LANE1), Note(NoteType.TAP, TapType.LANE8)), 0.0),
        Frame((Note(NoteType.HOLD, HoldData(TapType.LANE2, 0.04)),), 0.05),
    ])
    tracks = chart_to_tracks(chart, 5)
    assert tracks[10].tolist() == [2, 0, 0, 0]
    assert tracks[20].tolist() == [0, 1, 8, 0]
    restored = maidata_to_tracks(tracks_to_maidata(tracks, 5, difficulty=14.0), 5, len(tracks))
    assert np.array_equal(tracks, restored)
    song = SongData(
        Path("maidata.txt"), Path("track.mp3"), "test", "1",
        np.zeros((1030, 8), dtype=np.float32),
        (SongLevel(5, 13.7, np.zeros((1030, TRACK_COUNT), dtype=np.int32)),), "digest",
    )
    dataset = OverfitWindowDataset(song)
    assert len(dataset) == 103 and dataset[0]["features"].shape == (1024, 8)
    assert dataset[-1]["mask"].sum() == 10 and dataset[-1]["difficulty"].item() == np.float32(13.7)
    print("[dataset] 自检通过")


if __name__ == "__main__":
    _self_check()
    validate_all_songs()
