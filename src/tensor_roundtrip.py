"""独立验证 maidata 与计数、长按时长张量的往返转换。"""

import shutil

import numpy as np

from chart import (
    Chart, Frame, HoldData, Level, Note, NoteType, SlideSegment, SlideShape,
    TapType, TouchData, TouchType,
)
from config import CONFIG, ROOT_DIR
from maidata_parser import generate_maidata, parse_maidata


TAP_COUNT = 0
HOLD_START_COUNT = 1
HOLD_DURATION_1 = 2
HOLD_DURATION_2 = 3
TRACK_COUNT = 4


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
        start = round(frame.time_sec * rate)
        if start < 0:
            raise ValueError("音符时间不能为负")
        required_length = max(required_length, start + 1)
        for note in frame.notes:
            if note.type in (NoteType.TAP, NoteType.TOUCH):
                tap_events.append(start)
            elif note.type is NoteType.HOLD:
                duration = max(1, round(note.data.holdTime * rate))
                hold_events.append((start, duration))
                required_length = max(required_length, start + duration)
            elif note.type is NoteType.TOUCH_HOLD:
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


def tracks_to_chart(tracks: np.ndarray, level_idx: int = 5, title: str = "tensor-roundtrip", level_query: float = 0.0) -> Chart:
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
    chart.all_levels[level_idx] = Level("master", level_query, [
        Frame(tuple(notes), frame / rate) for frame, notes in sorted(grouped.items())
    ])
    return chart


def tracks_to_maidata(tracks: np.ndarray, level_idx: int = 5, title: str = "tensor-roundtrip", level_query: float = 0.0) -> str:
    return generate_maidata(tracks_to_chart(tracks, level_idx, title, level_query))


def txt2tensor2txt(text: str, level_idx: int = 5) -> tuple[np.ndarray, str]:
    tracks = maidata_to_tracks(text, level_idx)
    return tracks, tracks_to_maidata(tracks, level_idx)


def validate_all_songs() -> None:
    charts_dir = CONFIG.paths.charts_dir
    level_idx = CONFIG.inference.level_idx
    output_dir = ROOT_DIR / "tmp" / "txt2tensor2txt"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    paths = sorted(charts_dir.rglob("maidata.txt"))
    if not paths:
        raise ValueError(f"谱面目录中没有 maidata.txt: {charts_dir}")

    passed = 0
    generated_count = 0
    failed: list[str] = []
    for path in paths:
        relative = path.relative_to(charts_dir)
        try:
            source = path.read_text(encoding="utf-8")
            tracks, generated = txt2tensor2txt(source, level_idx)
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

    print(
        f"[tensor-roundtrip] 全曲验证完成: 总数={len(paths)} "
        f"生成={generated_count} 通过={passed} 整首跳过={len(failed)} "
        f"难度={level_idx} 输出={output_dir}"
    )
    report = ROOT_DIR / "tmp" / "tensor-roundtrip-failures.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(failed), encoding="utf-8")
    for error in failed[:20]:
        print(f"  {error}")
    if failed:
        print(f"[tensor-roundtrip] 完整失败清单: {report}")


def _self_check() -> None:
    rate = CONFIG.audio.frames_per_sec
    default_slide = Note(NoteType.SLIDE, [SlideSegment(
        SlideShape.Line, TapType.LANE4, TapType.LANE5,
        wait_duration=0.015, trace_duration=0.02,
    )])
    chart = Chart(title="test")
    chart.all_levels[5] = Level("master", 14, [
        Frame((
            Note(NoteType.TAP, TapType.LANE1),
            Note(NoteType.TOUCH, TouchData(TouchType.C)),
        ), 0.05),
        Frame((
            Note(NoteType.HOLD, HoldData(TapType.LANE2, 0.04)),
            Note(NoteType.TOUCH_HOLD, TouchData(TouchType.C, holdTime=0.06)),
        ), 0.10),
        Frame((default_slide,), 0.20),
    ])
    tracks = chart_to_tracks(chart, 5)
    assert tracks[round(0.05 * rate), TAP_COUNT] == 2
    hold_start = round(0.10 * rate)
    assert tracks[hold_start].tolist() == [0, 2, 8, 12]
    slide_start = round(0.20 * rate)
    assert tracks[slide_start].tolist() == [0, 1, 7, 0]
    restored = maidata_to_tracks(tracks_to_maidata(tracks), length=len(tracks))
    assert np.array_equal(tracks, restored)

    ignored_duration = np.zeros((1, TRACK_COUNT), dtype=np.int32)
    ignored_duration[0, HOLD_DURATION_1] = 10
    assert not tracks_to_chart(ignored_duration).all_levels[5].frames

    custom_slide = Note(NoteType.SLIDE, [SlideSegment(
        SlideShape.Line, TapType.LANE4, TapType.LANE5,
        wait_duration=0.015, trace_duration=0.02, is_default_wait=False,
    )])
    custom_chart = Chart(title="custom")
    custom_chart.all_levels[5] = Level("master", 14, [Frame((custom_slide,), 0.0)])
    try:
        chart_to_tracks(custom_chart, 5)
    except ValueError as error:
        assert "跳过整首歌" in str(error)
    else:
        raise AssertionError("非默认等待 SLIDE 必须跳过整首歌")
    print("[tensor-roundtrip] 自检通过")


if __name__ == "__main__":
    validate_all_songs()
