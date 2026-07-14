"""统计目标难度谱面的 Slide 等待、并发和活跃音符密度。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from statistics import mean, median

import numpy as np

from chart import Frame, Note, NoteType
from config import CONFIG
from maidata_parser import parse_maidata


RATE = 200


@dataclass
class SongMetrics:
    title: str
    level_idx: int
    level_query: float | None
    active_frames: int
    note_intervals_sec: list[float]
    type_means: dict[str, float]
    type_medians: dict[str, float]
    note_counts: dict[str, int]
    active_frames_by_type: dict[str, Counter[int]]
    start_counts: dict[str, list[int]]
    slide_frames: Counter[int]
    default_wait_segments: int
    custom_wait_segments: int


def _index(time_sec: float) -> int:
    return round(time_sec * RATE)


def _slide_intervals(note: Note, time_sec: float) -> list[tuple[int, int]]:
    """按多轨分支拆分 Slide；等待及轨迹阶段都属于活跃 Slide。"""
    branches: list[list] = []
    for segment in note.data:
        if not branches or segment.start_lane != branches[-1][-1].end_lane:
            branches.append([])
        branches[-1].append(segment)
    intervals = []
    for branch in branches:
        start_sec = time_sec
        end_sec = time_sec + sum(s.wait_duration + s.trace_duration for s in branch)
        start, end = _index(start_sec), _index(end_sec)
        if end > start:
            intervals.append((start, end))
    return intervals


def _add_interval(axis: np.ndarray, start: int, end: int, first: int) -> None:
    clipped_start, clipped_end = max(start, first), min(end, first + len(axis))
    if clipped_start < clipped_end:
        axis[clipped_start - first:clipped_end - first] += 1


def analyze_level(title: str, level_idx: int, level_query: float | None, frames: list[Frame]) -> SongMetrics:
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
    for frame_index, note, time_sec in events:
        if note.type in (NoteType.TAP, NoteType.TOUCH):
            axes["tap"][frame_index - first] += 1
            start_counts["tap"].append(frame_index)
            note_counts["tap"] += 1
        elif note.type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            _add_interval(axes["hold"], frame_index, _index(time_sec + note.data.holdTime), first)
            start_counts["hold"].append(frame_index)
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
            start_counts["slide"].append(frame_index)
            note_counts["slide"] += 1

    axes["total"] = axes["tap"] + axes["hold"] + axes["slide"]
    note_counts["total"] = sum(note_counts.values())
    starts_by_type: dict[str, list[int]] = {}
    for kind, indices in start_counts.items():
        counts = Counter(indices)
        starts_by_type[kind] = list(counts.values())
    note_times = sorted({frame.time_sec for frame in frames if frame.notes})
    return SongMetrics(
        title=title,
        level_idx=level_idx,
        level_query=level_query,
        active_frames=length,
        note_intervals_sec=[right - left for left, right in zip(note_times, note_times[1:])],
        type_means={name: float(axis.mean()) for name, axis in axes.items()},
        type_medians={name: float(np.median(axis)) for name, axis in axes.items()},
        note_counts=note_counts,
        active_frames_by_type={name: Counter(axis.tolist()) for name, axis in axes.items()},
        start_counts=starts_by_type,
        slide_frames=Counter(axes["slide"][axes["slide"] > 0].tolist()),
        default_wait_segments=default_wait_segments,
        custom_wait_segments=custom_wait_segments,
    )


def _format(value: float) -> str:
    return f"{value:.6f}"


def _print_note_count_groups(title: str, groups: dict[object, list[SongMetrics]], labels: dict[object, str] | None = None) -> None:
    print(f"\n{title}")
    for key, group in sorted(groups.items(), key=lambda item: (item[0] is None, item[0])):
        label = labels[key] if labels is not None else ("未知" if key is None else f"{key:g}")
        print(f"\n难度={label} 有效歌曲={len(group)}")
        for kind, note_label in (("tap", "Tap/Touch"), ("hold", "Hold/Touch Hold"), ("slide", "Slide"), ("total", "总音符")):
            counts = [song.note_counts[kind] for song in group]
            print(f"{note_label}: 平均={_format(mean(counts))} 中位数={_format(median(counts))}")
        intervals = [value for song in group for value in song.note_intervals_sec]
        if intervals:
            print(f"相邻音符间隔(秒): 平均={_format(mean(intervals))} 中位数={_format(median(intervals))}")
        else:
            print("相邻音符间隔(秒): 无相邻音符")


def _print_difficulty_metrics(all_level_songs: list[SongMetrics]) -> None:
    groups: dict[float | None, list[SongMetrics]] = defaultdict(list)
    for song in all_level_songs:
        groups[song.level_query].append(song)
    _print_note_count_groups("按浮点难度的单曲音符总出现次数统计", groups)

    # maidata 的 lv_1..lv_6 对应 Easy、Basic、Advanced、Expert、Master、Re:Master。
    names = {0: "未命名", 1: "Easy", 2: "Basic", 3: "Advanced", 4: "Expert", 5: "Master", 6: "Re:Master"}
    level_groups: dict[int, list[SongMetrics]] = defaultdict(list)
    for song in all_level_songs:
        level_groups[song.level_idx].append(song)
    _print_note_count_groups("按大难度的单曲音符总出现次数统计", level_groups, names)


def format_level_summary(items: list[tuple[float | None, list[Frame]]]) -> list[str]:
    """按浮点等级汇总生成谱面的音符数量和相邻音符间隔。"""
    groups: dict[float | None, list[tuple[int, list[float]]]] = defaultdict(list)
    for level_query, frames in items:
        note_count = sum(len(frame.notes) for frame in frames)
        times = sorted({frame.time_sec for frame in frames if frame.notes})
        groups[level_query].append((note_count, [right - left for left, right in zip(times, times[1:])]))
    lines = []
    for difficulty, songs in sorted(groups.items(), key=lambda item: (item[0] is None, item[0] or 0)):
        label = "未知" if difficulty is None else f"{difficulty:g}"
        counts = [count for count, _ in songs]
        intervals = [interval for _, values in songs for interval in values]
        lines.append(f"  浮点等级 {label}：这一组共有 {len(songs)} 首验证歌曲。")
        lines.append(f"    生成音符数量：每首平均生成 {mean(counts):.1f} 个；中位数是 {median(counts):.1f} 个。")
        if intervals:
            lines.append(f"    相邻音符间隔：相邻两个有音符时间点平均相隔 {mean(intervals):.4f} 秒；中位数相隔 {median(intervals):.4f} 秒。同一时刻的多押不算间隔。")
        else:
            lines.append("    相邻音符间隔：没有两个不同时间点的音符，因此暂时无法计算间隔。")
    return lines or ["  验证集中没有可用于生成统计的歌曲。"]


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


def _parse_one_level(text: str, level_idx: int):
    """隔离解析一个难度，避免其他难度的坏谱面丢弃其有效统计。"""
    filtered = re.sub(
        rf"^&inote_(?!{level_idx}=)[0-6]=.*?(?=^&|\Z)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return parse_maidata(filtered)


def report(charts_dir: Path = CONFIG.paths.charts_dir, level_idx: int = CONFIG.training.level_idx) -> None:
    songs: list[SongMetrics] = []
    all_level_songs: list[SongMetrics] = []
    missing_level = parse_errors = 0
    for path in sorted(charts_dir.rglob("maidata.txt")):
        try:
            text = path.read_text(encoding="utf-8")
            chart = parse_maidata(text)
            level = chart.all_levels[level_idx]
            if level is None:
                missing_level += 1
            else:
                songs.append(analyze_level(str(path.parent.relative_to(charts_dir)), level_idx, level.level_query, level.frames))
            for current_idx, current_level in enumerate(chart.all_levels):
                if current_level is not None and current_level.frames:
                    all_level_songs.append(analyze_level(
                        str(path.parent.relative_to(charts_dir)), current_idx, current_level.level_query, current_level.frames
                    ))
        except Exception as error:
            parse_errors += 1
            print(f"解析失败: {path.relative_to(charts_dir)}: {error}")
            # 全曲解析失败时，仍尝试收集其他不受损坏区块影响的大难度谱面。
            for current_idx in range(7):
                try:
                    isolated = _parse_one_level(text, current_idx)
                    current_level = isolated.all_levels[current_idx]
                    if current_level is not None and current_level.frames:
                        all_level_songs.append(analyze_level(
                            str(path.parent.relative_to(charts_dir)), current_idx, current_level.level_query, current_level.frames
                        ))
                except Exception:
                    pass

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
    _print_global_active_metrics(songs)
    _print_difficulty_metrics(all_level_songs)


def _self_check() -> None:
    from chart import HoldData, Note, SlideSegment, SlideShape, TapType

    slide = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE1, TapType.LANE2, 0.5, 1.0)])
    custom = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE3, TapType.LANE4, 0.3, 0.5, False)])
    # 构造的长期音、默认等待和显式等待覆盖三种统计路径。
    frames = [Frame((Note(NoteType.HOLD, HoldData(TapType.LANE1, 1.0)), slide, custom), 0.0)]
    metrics = analyze_level("test", 5, 13.5, frames)
    assert metrics.default_wait_segments == 1 and metrics.custom_wait_segments == 1
    assert metrics.slide_frames[1] == 140 and metrics.slide_frames[2] == 160
    assert metrics.note_counts == {"tap": 0, "hold": 1, "slide": 2, "total": 3}
    assert metrics.type_means["hold"] > 0 and metrics.type_medians["total"] >= 1
    assert format_level_summary([(13.5, [Frame((Note(NoteType.TAP, TapType.LANE1),), 0.0), Frame((Note(NoteType.TAP, TapType.LANE2),), 0.5)])])[-1] == "    相邻音符间隔：相邻两个有音符时间点平均相隔 0.5000 秒；中位数相隔 0.5000 秒。同一时刻的多押不算间隔。"
    print("[chart-metrics] 自检通过")


if __name__ == "__main__":
    _self_check()
    report()
