"""全量歌曲索引、现场音频增强和四列逐帧监督轴。"""

from collections import Counter
from dataclasses import dataclass, field
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from chart import Chart, Frame, HoldData, Level, Note, NoteType, SlideSegment, SlideShape, TapType
from config import CONFIG, ROOT_DIR
from maidata_parser import generate_maidata, parse_maidata


TAP_COUNT = 0
HOLD_START_COUNT = 1
HOLD_DURATION_1 = 2
HOLD_DURATION_2 = 3
TRACK_COUNT = 4
WINDOW_FRAMES = CONFIG.training.window_frames
TRAIN_STRIDE = CONFIG.training.stride
MUSIC_DATA_URL = "https://www.diving-fish.com/api/maimaidxprober/music_data"
MUSIC_DATA_CACHE = ROOT_DIR / ".cache" / "diving-fish-music-data.json"
MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
    sample_rate=CONFIG.audio.sample_rate,
    n_fft=CONFIG.audio.n_fft,
    win_length=CONFIG.audio.win_length,
    hop_length=CONFIG.audio.hop_length,
    f_min=CONFIG.audio.f_min,
    f_max=CONFIG.audio.f_max,
    n_mels=CONFIG.audio.n_mels,
    power=2.0,
)


@dataclass(frozen=True)
class LevelRecord:
    level_idx: int
    difficulty: float


@dataclass(frozen=True)
class SongRecord:
    chart_path: Path
    audio_path: Path
    title: str
    music_id: str
    levels: tuple[LevelRecord, ...]
    estimated_frames: int = 1
    content_digest: str = ""


@dataclass(frozen=True)
class DatasetIndex:
    songs: tuple[SongRecord, ...]
    digest: str
    music_data_digest: str
    excluded: dict[str, int]
    label_normalization: dict[str, int] = field(default_factory=dict)


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
        for note in frame.notes:
            if note.type in (NoteType.TAP, NoteType.TOUCH):
                if start >= 0:
                    tap_events.append(start)
                    required_length = max(required_length, start + 1)
            elif note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                duration = max(1, round(note.data.holdTime * rate))
                end = start + duration
                if end > 0:
                    clipped_start = max(start, 0)
                    hold_events.append((clipped_start, end - clipped_start))
                    required_length = max(required_length, end)
            elif note.type is NoteType.SLIDE:
                for branch in _slide_branches(note):
                    if not all(segment.is_default_wait for segment in branch):
                        raise ValueError(f"{start / rate:.3f}s SLIDE 使用非默认等待")
                    trace_start = start + round(branch[0].wait_duration * rate)
                    duration = max(1, round(sum(
                        segment.trace_duration for segment in branch
                    ) * rate))
                    end = trace_start + duration
                    if end > 0:
                        clipped_start = max(trace_start, 0)
                        hold_events.append((clipped_start, end - clipped_start))
                        required_length = max(required_length, end)
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
            continue
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
    if (tracks[:, TAP_COUNT] > 2).any() or (tracks[:, HOLD_START_COUNT] > 2).any():
        raise ValueError("Tap 和 Hold Start Count 不能超过 2")
    hold_count = tracks[:, HOLD_START_COUNT]
    durations = tracks[:, HOLD_DURATION_1:]
    invalid_durations = (
        ((hold_count == 0) & (durations != 0).any(axis=1))
        | ((hold_count == 1) & ((durations[:, 0] == 0) | (durations[:, 1] != 0)))
        | ((hold_count == 2) & (durations == 0).any(axis=1))
    )
    if invalid_durations.any():
        raise ValueError("Hold 数量和持续时长槽位不一致")
    rate = CONFIG.audio.frames_per_sec
    grouped: dict[int, list[Note]] = {}
    for frame, values in enumerate(tracks.astype(np.int64, copy=False)):
        for index in range(values[TAP_COUNT]):
            grouped.setdefault(frame, []).append(Note(
                NoteType.TAP, (TapType.LANE1, TapType.LANE8)[index],
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
    tracks[:, TAP_COUNT] = np.minimum(tracks[:, TAP_COUNT], 2)
    return tracks, tracks_to_maidata(tracks, level_idx)


def validate_all_songs() -> None:
    charts_dir = CONFIG.paths.charts_dir
    level_idx = CONFIG.inference.level_idx
    output_dir = ROOT_DIR / "tmp" / "txt2tensor2txt"
    report = ROOT_DIR / "tmp" / "tensor-roundtrip-failures.txt"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    paths = sorted(charts_dir.rglob("maidata.txt"))
    if not paths:
        raise ValueError(f"谱面目录中没有 maidata.txt: {charts_dir}")

    passed = 0
    generated_count = 0
    skipped: list[str] = []
    failed: list[str] = []
    for path in tqdm(paths, desc="全曲往返验证", disable=len(paths) <= 500):
        relative = path.relative_to(charts_dir)
        try:
            source = path.read_text(encoding="utf-8")
            tracks = maidata_to_tracks(source, level_idx)
            tracks[:, TAP_COUNT] = np.minimum(tracks[:, TAP_COUNT], 2)
        except Exception as error:
            skipped.append(f"{relative.parent}: {error}")
            continue
        try:
            generated = tracks_to_maidata(tracks, level_idx)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(generated, encoding="utf-8")
            generated_count += 1
            restored = maidata_to_tracks(generated, level_idx, len(tracks))
            if not np.array_equal(tracks, restored):
                raise ValueError("txt -> tensor -> txt -> tensor 不一致")
            passed += 1
        except Exception as error:
            failed.append(f"{relative.parent}: {error}")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(failed), encoding="utf-8")
    print(
        f"[dataset] 全曲往返验证完成: 总数={len(paths)} 生成={generated_count} "
        f"通过={passed} 输入跳过={len(skipped)} 失败={len(failed)} "
        f"难度={level_idx} 输出={output_dir}"
    )
    for error in skipped[:20]:
        print(f"  [输入跳过] {error}")
    for error in failed[:20]:
        print(f"  [失败] {error}")
    if failed:
        print(f"[dataset] 完整失败清单: {report}")
        raise AssertionError(f"有 {len(failed)} 首歌曲未通过张量往返验证")


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


def prepare_waveform(path: str | Path) -> torch.Tensor:
    waveform, sample_rate = load_audio(path)
    waveform = waveform.mean(dim=0, keepdim=True).float()
    if sample_rate != CONFIG.audio.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, CONFIG.audio.sample_rate)
    return waveform


def waveform_to_log_mel(waveform: torch.Tensor) -> np.ndarray:
    mel = MEL_TRANSFORM(waveform)
    return torch.log(mel.clamp_min(1e-10)).squeeze(0).transpose(0, 1).contiguous().numpy()


def extract_log_mel(path: str | Path) -> np.ndarray:
    return waveform_to_log_mel(prepare_waveform(path))


def _validate_music_data(data: object) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("歌曲数据必须是非空数组")
    for index, song in enumerate(tqdm(
        data, desc="校验歌曲元数据", disable=len(data) <= 500, leave=False,
    )):
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


def _fallback_music_id(title: str, chart_path: Path | None = None) -> str:
    source = f"{title} {chart_path.parent.name if chart_path is not None else ''}"
    suffix = re.search(r"\[(DX|ST)\]", source, re.IGNORECASE)
    kind = suffix.group(1).upper().replace("ST", "SD") if suffix else "SD"
    return f"fallback:{kind.lower()}:{_normalize_title(title)}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _audio_feature_frames(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"无法读取音频时长: {path}: {message}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"音频时长无效: {path}: {duration!r}")
    samples = round(duration * CONFIG.audio.sample_rate)
    return samples // CONFIG.audio.hop_length + 1


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


def build_dataset_index(charts_dir: Path = CONFIG.paths.charts_dir) -> DatasetIndex:
    music_data, music_digest = load_music_data()
    songs: list[SongRecord] = []
    excluded: Counter[str] = Counter()
    label_normalization: Counter[str] = Counter()
    maximum_duration = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    chart_paths = sorted(charts_dir.rglob("maidata.txt"))
    for chart_path in tqdm(
        chart_paths, desc="建立数据索引", disable=len(chart_paths) <= 500,
    ):
        audio_path = chart_path.parent / "track.mp3"
        if not audio_path.is_file():
            excluded["缺少音频"] += 1
            continue
        try:
            chart_bytes = chart_path.read_bytes()
            text = chart_bytes.decode("utf-8")
            chart = parse_maidata(text)
        except Exception:
            excluded["谱面解析失败"] += 1
            continue
        try:
            estimated_frames = _audio_feature_frames(audio_path)
            content_digest = hashlib.sha256(
                chart_bytes + _file_digest(audio_path).encode(),
            ).hexdigest()
        except Exception:
            excluded["音频读取失败"] += 1
            continue
        song = match_music(text, chart_path, music_data)
        title = str(song["title"]) if song is not None else chart.title
        music_id = str(song["id"]) if song is not None else _fallback_music_id(title, chart_path)
        levels: list[LevelRecord] = []
        for level_idx in range(2, 7):
            level = chart.all_levels[level_idx]
            if level is None or not level.frames:
                continue
            difficulty = (
                float(song["ds"][level_idx - 2])
                if song is not None and level_idx - 2 < len(song["ds"])
                else level.level_query
            )
            if difficulty is None or not 0 < difficulty <= 15:
                excluded["难度缺少有效定数"] += 1
                continue
            try:
                tracks = chart_to_tracks(chart, level_idx)
                level_stats: Counter[str] = Counter()
                tracks = normalize_tracks(tracks, estimated_frames, level_stats)
                if tracks[:, HOLD_DURATION_1:].max(initial=0) > maximum_duration:
                    raise ValueError("持续音超过配置上限")
                label_normalization.update(level_stats)
                levels.append(LevelRecord(level_idx, float(difficulty)))
            except Exception as error:
                message = str(error)
                if "SLIDE 使用非默认等待" in message:
                    reason = "难度跳过: SLIDE 使用非默认等待"
                elif "持续音超过配置上限" in message:
                    reason = "难度跳过: 持续音超过配置上限"
                else:
                    reason = f"难度跳过: {type(error).__name__}"
                excluded[reason] += 1
        if levels:
            songs.append(SongRecord(
                chart_path, audio_path, title, music_id, tuple(levels),
                estimated_frames, content_digest,
            ))
        else:
            excluded["没有可用难度"] += 1
    if not songs:
        raise ValueError("没有可用于训练的歌曲")
    summary = [
        (song.music_id, str(song.chart_path.relative_to(charts_dir)), song.estimated_frames,
         song.content_digest, [(level.level_idx, level.difficulty) for level in song.levels])
        for song in songs
    ]
    index_digest = hashlib.sha256(json.dumps(
        summary, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return DatasetIndex(
        tuple(songs), index_digest, music_digest, dict(excluded), dict(label_normalization),
    )


def _group_songs(index: DatasetIndex) -> list[tuple[str, list[SongRecord]]]:
    grouped: dict[str, list[SongRecord]] = {}
    for song in index.songs:
        grouped.setdefault(song.music_id, []).append(song)
    return sorted(grouped.items())


def _take_grouped_songs(
    groups: list[tuple[str, list[SongRecord]]], target: int,
) -> tuple[list[SongRecord], list[tuple[str, list[SongRecord]]]]:
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_music_id, songs) in enumerate(groups):
        for count, selected in tuple(reachable.items()):
            new_count = count + len(songs)
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = selected + (index,)
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError(f"无法在不拆分同曲版本的前提下选出恰好 {target} 首歌曲")
    selected_indexes = set(reachable[target])
    selected = [song for index in sorted(selected_indexes) for song in groups[index][1]]
    remaining = [group for index, group in enumerate(groups) if index not in selected_indexes]
    return selected, remaining


def split_songs(
    index: DatasetIndex, train_limit: int = CONFIG.training.song_limit,
) -> dict[str, tuple[SongRecord, ...]]:
    grouped_result: dict[str, list[tuple[str, list[SongRecord]]]] = {
        "train": [], "validation": [], "test": [],
    }
    for music_id, songs in _group_songs(index):
        digest = hashlib.sha256(f"{CONFIG.training.seed}:{music_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < CONFIG.training.test_ratio:
            name = "test"
        elif value < CONFIG.training.test_ratio + CONFIG.training.val_ratio:
            name = "validation"
        else:
            name = "train"
        grouped_result[name].append((music_id, songs))
    for groups in grouped_result.values():
        groups.sort(key=lambda item: hashlib.sha256(
            f"{CONFIG.training.seed}:sample:{item[0]}".encode(),
        ).digest())
    if train_limit:
        train_ratio = 1 - CONFIG.training.val_ratio - CONFIG.training.test_ratio
        targets = {
            "train": train_limit,
            "validation": round(train_limit * CONFIG.training.val_ratio / train_ratio),
            "test": round(train_limit * CONFIG.training.test_ratio / train_ratio),
        }
        return {
            name: tuple(_take_grouped_songs(grouped_result[name], target)[0])
            for name, target in targets.items()
        }
    return {
        name: tuple(song for _music_id, songs in groups for song in songs)
        for name, groups in grouped_result.items()
    }


def load_song(record: SongRecord) -> tuple[Chart, torch.Tensor]:
    chart = parse_maidata(record.chart_path.read_text(encoding="utf-8"))
    return chart, prepare_waveform(record.audio_path)


def normalize_tracks(
    tracks: np.ndarray, length: int, stats: Counter[str] | None = None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("标签长度必须大于 0")
    if stats is not None and len(tracks) > length:
        stats["Tap音频外忽略音符"] += int(tracks[length:, TAP_COUNT].sum())
        stats["Hold音频外忽略音符"] += int(tracks[length:, HOLD_START_COUNT].sum())
    tracks = tracks[:length].copy()
    if len(tracks) < length:
        tracks = np.pad(tracks, ((0, length - len(tracks)), (0, 0)))
    tap_overflow = np.maximum(tracks[:, TAP_COUNT] - 2, 0)
    if stats is not None:
        stats["Tap同帧超过2忽略音符"] += int(tap_overflow.sum())
    tracks[:, TAP_COUNT] = np.minimum(tracks[:, TAP_COUNT], 2)

    gap = CONFIG.inference.short_min_gap_frames
    if gap:
        for column, name in ((TAP_COUNT, "Tap"), (HOLD_START_COUNT, "Hold")):
            last_kept = -gap
            for frame_index in np.flatnonzero(tracks[:, column]):
                if frame_index - last_kept >= gap:
                    last_kept = int(frame_index)
                    continue
                count = int(tracks[frame_index, column])
                if stats is not None:
                    stats[f"{name}间隔不足忽略时间点"] += 1
                    stats[f"{name}间隔不足忽略音符"] += count
                if column == TAP_COUNT:
                    tracks[frame_index, TAP_COUNT] = 0
                else:
                    tracks[frame_index, HOLD_START_COUNT:] = 0

    for frame_index in np.flatnonzero(tracks[:, HOLD_START_COUNT]):
        count = int(tracks[frame_index, HOLD_START_COUNT])
        remaining = length - int(frame_index)
        for slot in range(count):
            column = HOLD_DURATION_1 + slot
            duration = int(tracks[frame_index, column])
            if duration > remaining:
                if stats is not None:
                    stats["Hold音频尾部截断音符"] += 1
                    stats["Hold音频尾部截断帧数"] += duration - remaining
                tracks[frame_index, column] = remaining
    return tracks


def _fit_tracks(tracks: np.ndarray, length: int) -> np.ndarray:
    return normalize_tracks(tracks, length)


def _resize_tracks(tracks: np.ndarray, speed: float, length: int) -> np.ndarray:
    result = np.zeros((length, TRACK_COUNT), dtype=np.int32)
    for source in np.flatnonzero(tracks[:, TAP_COUNT] | tracks[:, HOLD_START_COUNT]):
        target = round(source / speed)
        if target >= length:
            continue
        result[target, TAP_COUNT] = min(2, result[target, TAP_COUNT] + int(tracks[source, TAP_COUNT]))
        for slot in range(int(tracks[source, HOLD_START_COUNT])):
            target_slot = int(result[target, HOLD_START_COUNT])
            if target_slot == 2:
                break
            result[target, HOLD_START_COUNT] += 1
            result[target, HOLD_DURATION_1 + target_slot] = max(
                1, round(int(tracks[source, HOLD_DURATION_1 + slot]) / speed),
            )
    return normalize_tracks(result, length)


def _augment_waveform(waveform: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, float]:
    augmentation = CONFIG.augmentation
    speed = 1.0
    if rng.random() < augmentation.speed_probability:
        speed = float(rng.uniform(augmentation.min_speed, augmentation.max_speed))
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=max(1, round(waveform.shape[-1] / speed)),
            mode="linear", align_corners=False,
        ).squeeze(0)
    if rng.random() < augmentation.noise_probability and augmentation.max_noise_std:
        waveform = waveform + torch.randn_like(waveform) * augmentation.max_noise_std
    return waveform, speed


def _augment_features(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = features.copy()
    augmentation = CONFIG.augmentation
    if rng.random() < augmentation.pitch_probability and augmentation.max_pitch_steps:
        steps = float(rng.uniform(-augmentation.max_pitch_steps, augmentation.max_pitch_steps))
        source = np.arange(result.shape[1], dtype=np.float32) / (2 ** (steps / 12))
        source = np.clip(source, 0, result.shape[1] - 1)
        left = np.floor(source).astype(np.int64)
        right = np.minimum(left + 1, result.shape[1] - 1)
        weight = (source - left).astype(np.float32)
        result = result[:, left] * (1 - weight) + result[:, right] * weight
    if rng.random() < augmentation.frequency_mask_probability and augmentation.max_frequency_mask_bins:
        width = int(rng.integers(1, augmentation.max_frequency_mask_bins + 1))
        start = int(rng.integers(0, result.shape[1] - width + 1))
        result[:, start:start + width] = result.mean()
    if rng.random() < augmentation.eq_probability and augmentation.max_eq_gain:
        gains = rng.uniform(-augmentation.max_eq_gain, augmentation.max_eq_gain, 8)
        curve = np.interp(np.linspace(0, 7, result.shape[1]), np.arange(8), gains)
        result += curve.astype(np.float32)
    return result


def _window(features: np.ndarray, events: np.ndarray, start: int) -> dict[str, torch.Tensor]:
    end = min(start + WINDOW_FRAMES, len(features))
    length = end - start
    feature_window = np.zeros((WINDOW_FRAMES, features.shape[1]), dtype=np.float32)
    event_window = np.zeros((WINDOW_FRAMES, TRACK_COUNT), dtype=np.int64)
    mask = np.zeros(WINDOW_FRAMES, dtype=bool)
    feature_window[:length] = features[start:end]
    event_window[:length] = events[start:end]
    mask[:length] = True
    return {
        "features": torch.from_numpy(feature_window),
        "events": torch.from_numpy(event_window),
        "mask": torch.from_numpy(mask),
    }


def _shift_window(sample: dict[str, torch.Tensor], shift: int) -> None:
    if not shift:
        return
    for key in ("features", "events", "mask"):
        sample[key] = torch.roll(sample[key], shift, 0)
        boundary = slice(0, shift) if shift > 0 else slice(shift, None)
        sample[key][boundary] = 0


def _stack_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def window_starts(length: int, offset: int, training: bool) -> list[int]:
    if length <= 0:
        return []
    if length <= WINDOW_FRAMES:
        return [0]
    last = length - WINDOW_FRAMES
    starts = set(range(offset if training else 0, last + 1, TRAIN_STRIDE))
    starts.add(0)
    starts.add(last)
    return sorted(starts)


def _produce_window_batches(songs: tuple[SongRecord, ...], epoch: int, training: bool):
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    worker_count = worker.num_workers if worker is not None else 1
    base_rng = np.random.default_rng(CONFIG.training.seed + epoch)
    order = base_rng.permutation(len(songs)) if training else np.arange(len(songs))
    order = order[worker_id::worker_count]
    batch: list[dict[str, torch.Tensor]] = []
    shuffle_buffer: list[dict[str, torch.Tensor]] = []
    offset = int(base_rng.integers(0, TRAIN_STRIDE)) if training else 0
    rng = np.random.default_rng(CONFIG.training.seed + epoch * 1009 + worker_id)
    for song_index in order:
        record = songs[int(song_index)]
        chart, waveform = load_song(record)
        speed = 1.0
        if training:
            waveform, speed = _augment_waveform(waveform, rng)
        features = waveform_to_log_mel(waveform)
        if training:
            features = _augment_features(features, rng)
        for level in record.levels:
            tracks = chart_to_tracks(chart, level.level_idx)
            tracks = _resize_tracks(tracks, speed, len(features)) if speed != 1.0 else _fit_tracks(tracks, len(features))
            starts = window_starts(len(features), offset, training)
            if training and CONFIG.training.event_window_ratio and starts:
                count = round(len(starts) * CONFIG.training.event_window_ratio)
                event_frames = np.flatnonzero(tracks[:, TAP_COUNT] | tracks[:, HOLD_START_COUNT])
                if event_frames.size and count:
                    centers = rng.choice(event_frames, count, replace=event_frames.size < count)
                    starts.extend(max(0, min(
                        max(0, len(features) - WINDOW_FRAMES), int(x) - WINDOW_FRAMES // 2,
                    )) for x in centers)
                    starts = list(set(starts))
            if training:
                rng.shuffle(starts)
            for start in starts:
                sample = _window(features, tracks, start)
                if training and rng.random() < CONFIG.augmentation.shift_probability:
                    maximum = round(CONFIG.augmentation.max_shift_sec * CONFIG.audio.frames_per_sec)
                    _shift_window(sample, int(rng.integers(-maximum, maximum + 1)) if maximum else 0)
                sample["difficulty"] = torch.tensor(level.difficulty, dtype=torch.float32)
                if training:
                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) < CONFIG.training.batch_size * 4:
                        continue
                    indexes = rng.choice(
                        len(shuffle_buffer), CONFIG.training.batch_size, replace=False,
                    )
                    selected = set(int(index) for index in indexes)
                    yield _stack_batch([shuffle_buffer[index] for index in indexes])
                    shuffle_buffer = [
                        item for index, item in enumerate(shuffle_buffer) if index not in selected
                    ]
                else:
                    batch.append(sample)
                    if len(batch) == CONFIG.training.batch_size:
                        yield _stack_batch(batch)
                        batch.clear()
    if training:
        rng.shuffle(shuffle_buffer)
        batch.extend(shuffle_buffer)
        while len(batch) >= CONFIG.training.batch_size:
            yield _stack_batch(batch[:CONFIG.training.batch_size])
            del batch[:CONFIG.training.batch_size]
    if batch:
        yield _stack_batch(batch)


class WindowBatchDataset(IterableDataset):
    def __init__(self, songs: tuple[SongRecord, ...], epoch: int, training: bool):
        self.songs = songs
        self.epoch = epoch
        self.training = training

    def __iter__(self):
        return _produce_window_batches(self.songs, self.epoch, self.training)


def _init_worker(_worker_id: int) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // CONFIG.training.num_workers))


def iter_window_batches(songs: tuple[SongRecord, ...], epoch: int, training: bool):
    workers = CONFIG.training.num_workers
    options = {
        "batch_size": None,
        "num_workers": workers,
        "pin_memory": CONFIG.training.pin_memory and torch.cuda.is_available(),
    }
    if workers:
        options.update({
            "prefetch_factor": CONFIG.training.prefetch_factor,
            "persistent_workers": False,
            "worker_init_fn": _init_worker,
        })
    return DataLoader(WindowBatchDataset(songs, epoch, training), **options)


def load_song_data(record: SongRecord) -> tuple[np.ndarray, tuple[tuple[LevelRecord, np.ndarray], ...]]:
    chart, waveform = load_song(record)
    features = waveform_to_log_mel(waveform)
    levels = []
    for level in record.levels:
        tracks = _fit_tracks(chart_to_tracks(chart, level.level_idx), len(features))
        levels.append((level, tracks))
    return features, tuple(levels)


def iter_loaded_window_batches(loaded_songs):
    batch: list[dict[str, torch.Tensor]] = []
    for _record, features, levels in loaded_songs:
        starts = window_starts(len(features), 0, False)
        for level, tracks in levels:
            for start in starts:
                sample = _window(features, tracks, start)
                sample["difficulty"] = torch.tensor(level.difficulty, dtype=torch.float32)
                batch.append(sample)
                if len(batch) == CONFIG.training.batch_size:
                    yield _stack_batch(batch)
                    batch.clear()
    if batch:
        yield _stack_batch(batch)


def _self_check() -> None:
    chart = Chart(title="test", first_sec=0.05)
    slide = Note(NoteType.SLIDE, [
        SlideSegment(SlideShape.Line, TapType.LANE1, TapType.LANE2, 0.5, 1.0),
    ])
    chart.all_levels[5] = Level("master", 14.0, [
        Frame((Note(NoteType.TAP, TapType.LANE1), Note(NoteType.TAP, TapType.LANE8), Note(NoteType.TAP, TapType.LANE3)), 0.0),
        Frame((Note(NoteType.HOLD, HoldData(TapType.LANE2, 0.04)),), 0.05),
        Frame((slide,), 0.1),
    ])
    tracks = chart_to_tracks(chart, 5)
    assert tracks[10].tolist() == [3, 0, 0, 0]
    assert tracks[20].tolist() == [0, 1, 8, 0]
    assert tracks[130, HOLD_DURATION_1] == 200
    chart.all_levels[5].frames.append(Frame((
        Note(NoteType.HOLD, HoldData(TapType.LANE1, 1.0)),
        Note(NoteType.HOLD, HoldData(TapType.LANE2, 1.0)),
        Note(NoteType.HOLD, HoldData(TapType.LANE3, 1.0)),
    ), 2.0))
    assert chart_to_tracks(chart, 5)[410, HOLD_START_COUNT] == 2
    resized = _resize_tracks(tracks, 2.0, len(tracks) // 2)
    assert resized[5, TAP_COUNT] == 2 and resized[65, HOLD_DURATION_1] == 100

    aligned = np.zeros((32, TRACK_COUNT), dtype=np.int32)
    aligned[4] = (1, 1, 9, 0)
    aligned[10] = (1, 1, 7, 0)
    aligned[20] = (2, 2, 11, 13)
    for speed in (0.98, 1.0, 1.02):
        length = round(len(aligned) / speed)
        scaled = _resize_tracks(aligned, speed, length)
        assert scaled[round(10 / speed), HOLD_DURATION_1] == max(1, round(7 / speed))
        target = round(20 / speed)
        assert scaled[round(20 / speed)].tolist() == [
            2, 2,
            min(max(1, round(11 / speed)), length - target),
            min(max(1, round(13 / speed)), length - target),
        ]

    class _SpeedRng:
        def random(self):
            return 0.0

        def uniform(self, _minimum, _maximum):
            return 1.02

    waveform = torch.arange(320, dtype=torch.float32).unsqueeze(0)
    augmented, speed = _augment_waveform(waveform, _SpeedRng())
    scaled = _resize_tracks(aligned, speed, round(len(aligned) / speed))
    assert augmented.shape[-1] == round(waveform.shape[-1] / speed)
    assert np.flatnonzero(scaled[:, TAP_COUNT]).tolist() == sorted({
        round(frame / speed) for frame in np.flatnonzero(aligned[:, TAP_COUNT])
    })
    close = np.zeros((32, TRACK_COUNT), dtype=np.int32)
    close[4] = (1, 1, 9, 0)
    close[5] = (2, 2, 7, 8)
    stats: Counter[str] = Counter()
    normalized = normalize_tracks(close, len(close), stats)
    assert normalized[4].tolist() == [1, 1, 9, 0] and not normalized[5].any()
    assert stats["Tap间隔不足忽略音符"] == 2
    assert stats["Hold间隔不足忽略音符"] == 2
    tail = np.zeros((12, TRACK_COUNT), dtype=np.int32)
    tail[1] = (0, 1, 100, 0)
    tail[11] = (0, 1, 10, 0)
    stats.clear()
    tail = normalize_tracks(tail, len(tail), stats)
    assert tail[1, HOLD_DURATION_1] == 11 and tail[11, HOLD_DURATION_1] == 1
    assert stats["Hold音频尾部截断音符"] == 2

    sample = {
        "features": torch.arange(8, dtype=torch.float32).unsqueeze(1),
        "events": torch.arange(8, dtype=torch.int64).unsqueeze(1).repeat(1, TRACK_COUNT),
        "mask": torch.ones(8, dtype=torch.bool),
    }
    original = {key: value.clone() for key, value in sample.items()}
    _shift_window(sample, 3)
    for key in sample:
        assert torch.equal(sample[key][3:], original[key][:-3])
        assert not sample[key][:3].any()
    sample = {key: value.clone() for key, value in original.items()}
    _shift_window(sample, -2)
    for key in sample:
        assert torch.equal(sample[key][:-2], original[key][2:])
        assert not sample[key][-2:].any()

    canonical = np.zeros((41, TRACK_COUNT), dtype=np.int32)
    canonical[0] = (1, 0, 0, 0)
    canonical[5] = (2, 0, 0, 0)
    canonical[12] = (0, 1, 3, 0)
    canonical[24] = (1, 2, 7, 11)
    canonical[40] = (2, 1, 1, 0)
    generated = tracks_to_maidata(canonical, 5, "roundtrip", 14.0)
    restored = maidata_to_tracks(generated, 5, len(canonical))
    assert np.array_equal(canonical, restored)
    source = "&title=source\n&artist=test\n&first=0.005\n&lv_5=14\n&inote_5=(12000){4}1/8,2h[4:1],E"
    source_tracks, source_generated = txt2tensor2txt(source, 5)
    assert np.array_equal(
        source_tracks, maidata_to_tracks(source_generated, 5, len(source_tracks)),
    )
    invalid = canonical.copy()
    invalid[1, HOLD_DURATION_1] = 2
    try:
        tracks_to_chart(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("非规范 Hold 时长槽位必须被拒绝")
    first = SongRecord(Path("a"), Path("b"), "A", "same", (LevelRecord(5, 13.0),))
    second = SongRecord(Path("c"), Path("d"), "A", "same", (LevelRecord(5, 13.0),))
    split = split_songs(DatasetIndex((first, second), "x", "y", {}), 0)
    assert sum(len(value) for value in split.values()) == 2
    assert any(len(value) == 2 for value in split.values())
    records = tuple(
        SongRecord(Path(str(index)), Path("b"), str(index), str(index), (LevelRecord(5, 13.0),))
        for index in range(40)
    )
    limited = split_songs(DatasetIndex(records, "x", "y", {}), 10)
    assert {name: len(value) for name, value in limited.items()} == {
        "train": 10, "validation": 2, "test": 1,
    }
    assert _fallback_music_id("test [DX]") == "fallback:dx:test"
    assert _fallback_music_id("test", Path("test [DX]/maidata.txt")) == "fallback:dx:test"
    assert window_starts(1, 0, False) == [0]
    assert window_starts(WINDOW_FRAMES, 0, False) == [0]
    assert window_starts(WINDOW_FRAMES + 1, 0, False) == [0, 1]
    assert window_starts(1536, 0, False) == [0, 512]
    starts = window_starts(3000, 511, True)
    assert starts == [0, 511, 1023, 1535, 1976]
    negative = Chart(first_sec=-0.1)
    negative.all_levels[5] = Level("test", 1.0, [
        Frame((Note(NoteType.TAP, TapType.LANE1),), 0.0),
        Frame((Note(NoteType.HOLD, HoldData(TapType.LANE1, 0.2)),), 0.0),
    ])
    clipped = chart_to_tracks(negative, 5)
    assert clipped[0].tolist() == [0, 1, 20, 0]
    print("[dataset] 自检通过")


if __name__ == "__main__":
    validate_all_songs()
