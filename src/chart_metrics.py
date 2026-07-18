"""统计目标难度谱面的 Slide 等待、并发和活跃音符密度。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np

from chart import Frame, Note, NoteType, SlideSegment
from config import CONFIG
from maidata_parser import parse_maidata


RATE = CONFIG.audio.frames_per_sec
MIN_GAP = CONFIG.inference.short_min_gap_frames


@dataclass
class SongMetrics:
    title: str
    level_query: float | None
    active_frames: int
    type_means: dict[str, float]
    type_medians: dict[str, float]
    note_counts: dict[str, int]
    active_frames_by_type: dict[str, Counter[int]]
    start_counts: dict[str, list[int]]
    slide_frames: Counter[int]
    default_wait_segments: int
    custom_wait_segments: int
    label_gaps: dict[str, list[int]]
    close_start_points: dict[str, int]
    close_note_counts: dict[str, int]
    closest_starts: dict[str, tuple[int, int] | None]
    label_note_counts: dict[str, int]


def _index(time_sec: float) -> int:
    return round(time_sec * RATE)


def _slide_branches(note: Note) -> list[list[SlideSegment]]:
    branches: list[list[SlideSegment]] = []
    for segment in note.data:
        if not branches or segment.starts_new_branch:
            branches.append([])
        branches[-1].append(segment)
    return branches


def _slide_branch_timing(branch: list[SlideSegment]) -> tuple[float, float]:
    timed_segment = next(
        (segment for segment in branch if segment.trace_duration > 0), branch[0],
    )
    return timed_segment.wait_duration, sum(segment.trace_duration for segment in branch)


def _slide_intervals(note: Note, time_sec: float) -> list[tuple[int, int]]:
    """按多轨分支拆分 Slide；等待及轨迹阶段都属于活跃 Slide。"""
    intervals = []
    for branch in _slide_branches(note):
        wait_duration, trace_duration = _slide_branch_timing(branch)
        start_sec = time_sec
        end_sec = time_sec + wait_duration + trace_duration
        start, end = _index(start_sec), _index(end_sec)
        if end > start:
            intervals.append((start, end))
    return intervals


def _slide_trace_starts(note: Note, time_sec: float) -> list[int]:
    return [
        _index(time_sec + _slide_branch_timing(branch)[0])
        for branch in _slide_branches(note)
    ]


def _add_interval(axis: np.ndarray, start: int, end: int, first: int) -> None:
    clipped_start, clipped_end = max(start, first), min(end, first + len(axis))
    if clipped_start < clipped_end:
        axis[clipped_start - first:clipped_end - first] += 1


def analyze_level(title: str, level_query: float | None, frames: list[Frame]) -> SongMetrics:
    if not frames:
        raise ValueError("谱面没有音符")
    first = min(_index(frame.time_sec) for frame in frames)
    last = first
    events: list[tuple[int, Note, float]] = []
    for frame in frames:
        frame_index = _index(frame.time_sec)
        for note in frame.notes:
            events.append((frame_index, note, frame.time_sec))
            last = max(last, frame_index)
            if note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                last = max(last, _index(frame.time_sec + note.data.holdTime))
            elif note.type is NoteType.SLIDE:
                for _start, end in _slide_intervals(note, frame.time_sec):
                    last = max(last, end)

    # 末尾给瞬时音符保留一帧；持续区间右端保持半开，和数据集一致。
    length = max(1, last - first + 1)
    axes = {name: np.zeros(length, dtype=np.int16) for name in ("tap", "hold", "slide")}
    start_counts = {name: [] for name in ("tap", "hold", "slide")}
    note_counts = {name: 0 for name in ("tap", "hold", "slide")}
    default_wait_segments = custom_wait_segments = 0
    label_starts = {name: Counter() for name in ("tap", "hold")}
    for frame_index, note, time_sec in events:
        if note.type in (NoteType.TAP, NoteType.TOUCH):
            axes["tap"][frame_index - first] += 1
            start_counts["tap"].append(frame_index)
            label_starts["tap"][frame_index] += 1
            note_counts["tap"] += 1
        elif note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            _add_interval(axes["hold"], frame_index, _index(time_sec + note.data.holdTime), first)
            start_counts["hold"].append(frame_index)
            label_starts["hold"][frame_index] += 1
            note_counts["hold"] += 1
        elif note.type is NoteType.SLIDE:
            for segment in note.data:
                if segment.trace_duration <= 0:
                    continue
                if segment.is_default_wait:
                    default_wait_segments += 1
                else:
                    custom_wait_segments += 1
            for start, end in _slide_intervals(note, time_sec):
                _add_interval(axes["slide"], start, end, first)
            for start in _slide_trace_starts(note, time_sec):
                label_starts["hold"][start] += 1
            start_counts["slide"].append(frame_index)
            note_counts["slide"] += 1

    axes["total"] = axes["tap"] + axes["hold"] + axes["slide"]
    note_counts["total"] = sum(note_counts.values())
    starts_by_type: dict[str, list[int]] = {}
    for kind, indices in start_counts.items():
        counts = Counter(indices)
        starts_by_type[kind] = list(counts.values())
    label_gaps: dict[str, list[int]] = {}
    close_start_points: dict[str, int] = {}
    close_note_counts: dict[str, int] = {}
    closest_starts: dict[str, tuple[int, int] | None] = {}
    for kind, counts in label_starts.items():
        points = sorted(counts)
        label_gaps[kind] = [right - left for left, right in zip(points, points[1:])]
        kept: list[int] = []
        ignored_points = ignored_notes = 0
        for point in points:
            if kept and point - kept[-1] < MIN_GAP:
                ignored_points += 1
                ignored_notes += counts[point]
            else:
                kept.append(point)
        close_start_points[kind] = ignored_points
        close_note_counts[kind] = ignored_notes
        closest_starts[kind] = min(
            zip(points, points[1:]), key=lambda pair: pair[1] - pair[0], default=None,
        )
    return SongMetrics(
        title=title,
        level_query=level_query,
        active_frames=length,
        type_means={name: float(axis.mean()) for name, axis in axes.items()},
        type_medians={name: float(np.median(axis)) for name, axis in axes.items()},
        note_counts=note_counts,
        active_frames_by_type={name: Counter(axis.tolist()) for name, axis in axes.items()},
        start_counts=starts_by_type,
        slide_frames=Counter(axes["slide"][axes["slide"] > 0].tolist()),
        default_wait_segments=default_wait_segments,
        custom_wait_segments=custom_wait_segments,
        label_gaps=label_gaps,
        close_start_points=close_start_points,
        close_note_counts=close_note_counts,
        closest_starts=closest_starts,
        label_note_counts={kind: sum(counts.values()) for kind, counts in label_starts.items()},
    )


def _format(value: float) -> str:
    return f"{value:.6f}"


def _print_difficulty_metrics(songs: list[SongMetrics]) -> None:
    groups: dict[float | None, list[SongMetrics]] = defaultdict(list)
    for song in songs:
        groups[song.level_query].append(song)
    print("\n按浮点难度的单曲音符总出现次数统计")
    for difficulty, group in sorted(groups.items(), key=lambda item: (item[0] is None, item[0] or 0)):
        label = "未知" if difficulty is None else f"{difficulty:g}"
        print(f"\n浮点难度={label} 有效歌曲={len(group)}")
        for kind, label in (("tap", "Tap/Touch"), ("hold", "Hold/Touch Hold"), ("slide", "Slide"), ("total", "总音符")):
            counts = [song.note_counts[kind] for song in group]
            print(f"{label}: 平均={_format(mean(counts))} 中位数={_format(median(counts))}")


def _histogram_median(histogram: Counter[int]) -> float:
    total = sum(histogram.values())
    if not total:
        return 0.0
    left = (total - 1) // 2
    right = total // 2
    seen = 0
    values = []
    for value in sorted(histogram):
        seen += histogram[value]
        while len(values) < 2 and seen > (left if not values else right):
            values.append(value)
    return sum(values) / len(values)


def _print_global_active_metrics(songs: list[SongMetrics]) -> None:
    print("\n整首歌范围内的活跃音符统计")
    for kind, label in (("tap", "Tap/Touch"), ("hold", "Hold/Touch Hold"), ("slide", "Slide"), ("total", "总活跃音符")):
        histogram: Counter[int] = Counter()
        for song in songs:
            histogram.update(song.active_frames_by_type[kind])
        frames = sum(histogram.values())
        average = sum(value * count for value, count in histogram.items()) / frames if frames else 0.0
        print(f"{label}: 帧数={frames} 平均={_format(average)} 中位数={_format(_histogram_median(histogram))}")


def _print_new_start_metrics(songs: list[SongMetrics]) -> None:
    print("\n有新开始音符的时间点统计")
    for kind, label in (("tap", "Tap/Touch"), ("hold", "Hold/Touch Hold"), ("slide", "Slide")):
        counts = [value for song in songs for value in song.start_counts[kind]]
        if counts:
            print(f"{label}: 时间点数={len(counts)} 平均={_format(mean(counts))} 中位数={_format(median(counts))}")
        else:
            print(f"{label}: 时间点数=0")


def _print_label_gap_metrics(songs: list[SongMetrics]) -> None:
    print(f"\n模型标签相邻起点间隔统计（小于 {MIN_GAP} 帧将忽略后一个时间点）")
    for kind, label in (("tap", "Tap/Touch"), ("hold", "Hold/Touch Hold/Slide轨迹")):
        gaps = [gap for song in songs for gap in song.label_gaps[kind]]
        ignored_points = sum(song.close_start_points[kind] for song in songs)
        ignored_notes = sum(song.close_note_counts[kind] for song in songs)
        affected = [song for song in songs if song.close_start_points[kind]]
        total_notes = sum(song.label_note_counts[kind] for song in songs)
        if not gaps:
            print(f"{label}: 无相邻时间点")
            continue
        array = np.asarray(gaps, dtype=np.int64)
        closest_song = min(
            (song for song in songs if song.closest_starts[kind] is not None),
            key=lambda song: song.closest_starts[kind][1] - song.closest_starts[kind][0],
        )
        closest = closest_song.closest_starts[kind]
        percentiles = np.percentile(array, (1, 5, 95, 99))
        histogram = Counter(int(value) for value in array if 0 < value < MIN_GAP)
        print(
            f"{label}: 间隔数={len(gaps)} 平均={_format(float(array.mean()))} "
            f"中位数={_format(float(np.median(array)))} 最小={int(array.min())} 最大={int(array.max())} "
            f"P1/P5/P95/P99={'/'.join(_format(float(value)) for value in percentiles)}"
        )
        print(
            f"  将忽略时间点={ignored_points} 音符={ignored_notes}/{total_notes} "
            f"({ignored_notes / total_notes:.4%}) 受影响歌曲={len(affected)}/{len(songs)} "
            f"1-{MIN_GAP - 1}帧分布={dict(sorted(histogram.items()))}"
        )
        print(
            f"  极端样本={closest_song.title} 帧={closest[0]}->{closest[1]} "
            f"间隔={closest[1] - closest[0]}"
        )


def report(charts_dir: Path = CONFIG.paths.charts_dir, level_idx: int = CONFIG.inference.level_idx) -> None:
    songs: list[SongMetrics] = []
    missing_level = parse_errors = 0
    for path in sorted(charts_dir.rglob("maidata.txt")):
        try:
            chart = parse_maidata(path.read_text(encoding="utf-8"))
            level = chart.all_levels[level_idx]
            if level is None:
                missing_level += 1
                continue
            songs.append(analyze_level(str(path.parent.relative_to(charts_dir)), level.level_query, level.frames))
        except Exception as error:
            parse_errors += 1
            print(f"解析失败: {path.relative_to(charts_dir)}: {error}")

    slide_songs = [song for song in songs if song.slide_frames]
    default_songs = [song for song in songs if song.default_wait_segments]
    custom_songs = [song for song in songs if song.custom_wait_segments]
    default_segments = sum(song.default_wait_segments for song in songs)
    custom_segments = sum(song.custom_wait_segments for song in songs)
    print(f"有效歌曲={len(songs)} 含SLIDE歌曲={len(slide_songs)} 缺少难度 {level_idx}={missing_level} 解析失败={parse_errors}")
    if custom_segments:
        print(f"默认等待 : 非默认等待 = {default_segments / custom_segments:.2f} : 1")
    else:
        print(f"默认等待 : 非默认等待 = {default_segments} : 0")
    print(f"含默认等待 SLIDE={len(default_songs)} 首")
    print(f"含非默认等待 SLIDE={len(custom_songs)} 首，占有效歌曲={len(custom_songs) / len(songs):.2%}" if songs else "含非默认等待 SLIDE=0 首")

    slide_counts: Counter[int] = Counter()
    for song in songs:
        slide_counts.update(song.slide_frames)
    active_slide_frames = sum(slide_counts.values())
    if active_slide_frames:
        average = sum(count * frames for count, frames in slide_counts.items()) / active_slide_frames
        expanded = np.repeat(np.fromiter(slide_counts.keys(), dtype=np.int16), np.fromiter(slide_counts.values(), dtype=np.int64))
        maximum = max(slide_counts)
        maximum_song = next(song.title for song in songs if maximum in song.slide_frames)
        print(f"\nSLIDE活跃帧={active_slide_frames} 活跃秒数={active_slide_frames / RATE:.3f}")
        print(f"平均并发数={_format(average)}")
        print(f"中位并发数={float(np.median(expanded)):g}")
        print(f"最大并发数={maximum} 歌曲={maximum_song}")
        for count in sorted(slide_counts):
            frames = slide_counts[count]
            print(f"并发{count}={frames}帧 ({frames / active_slide_frames:.4%})")
    _print_new_start_metrics(songs)
    _print_label_gap_metrics(songs)
    _print_global_active_metrics(songs)
    _print_difficulty_metrics(songs)


def _self_check() -> None:
    from chart import HoldData, Note, SlideSegment, SlideShape, TapType

    slide = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE1, TapType.LANE2, 0.5, 1.0)])
    custom = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE3, TapType.LANE4, 0.3, 0.5, False)])
    # 构造的长期音、默认等待和显式等待覆盖三种统计路径。
    frames = [
        Frame((Note(NoteType.TAP, TapType.LANE1), Note(NoteType.HOLD, HoldData(TapType.LANE1, 1.0)), slide, custom), 0.0),
        Frame((Note(NoteType.TAP, TapType.LANE2), Note(NoteType.HOLD, HoldData(TapType.LANE2, 1.0))), 0.02),
    ]
    metrics = analyze_level("test", 13.5, frames)
    assert metrics.default_wait_segments == 1 and metrics.custom_wait_segments == 1
    assert metrics.slide_frames[1] == 140 and metrics.slide_frames[2] == 160
    assert metrics.note_counts == {"tap": 2, "hold": 2, "slide": 2, "total": 6}
    assert metrics.type_means["hold"] > 0 and metrics.type_medians["total"] >= 1
    assert metrics.close_start_points == {"tap": 1, "hold": 1}
    assert metrics.close_note_counts == {"tap": 1, "hold": 1}
    parallel = Note(NoteType.SLIDE, [
        SlideSegment(
            SlideShape.PP, TapType.LANE4, TapType.LANE4, 0.5, 0.0,
            starts_new_branch=True,
        ),
        SlideSegment(
            SlideShape.QQ, TapType.LANE4, TapType.LANE4, 0.3, 0.5,
            starts_new_branch=True,
        ),
    ])
    chained = Note(NoteType.SLIDE, [
        SlideSegment(
            SlideShape.Line, TapType.LANE1, TapType.LANE3,
            starts_new_branch=True,
        ),
        SlideSegment(SlideShape.Line, TapType.LANE3, TapType.LANE5, 0.4, 0.6),
    ])
    assert _slide_trace_starts(parallel, 1.0) == [_index(1.5), _index(1.3)]
    assert _slide_intervals(parallel, 1.0) == [
        (_index(1.0), _index(1.5)), (_index(1.0), _index(1.8)),
    ]
    assert _slide_trace_starts(chained, 1.0) == [_index(1.4)]
    assert _slide_intervals(chained, 1.0) == [(_index(1.0), _index(2.0))]
    print("[chart-metrics] 自检通过")


if __name__ == "__main__":
    _self_check()
    report()
