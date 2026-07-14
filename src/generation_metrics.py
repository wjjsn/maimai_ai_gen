"""比较正式生成事件、原谱事件和 Master 总体统计。"""

from statistics import mean, median

import numpy as np

from config import CONFIG
from tensor_roundtrip import HOLD_START_COUNT, TAP_COUNT


MASTER_MEAN = {"tap": 497.522670, "long": 138.738035, "total": 636.260705, "interval": 0.251188}
MASTER_MEDIAN = {"tap": 485.0, "long": 133.0, "total": 631.5, "interval": 0.188679}


def _ratio(value: float, reference: float) -> str:
    return f"{value / reference * 100:.1f}%" if reference else "无基准"


def _counts(tracks: np.ndarray) -> dict[str, int]:
    tap = int(tracks[:, TAP_COUNT].sum())
    long = int(tracks[:, HOLD_START_COUNT].sum())
    return {"tap": tap, "long": long, "total": tap + long}


def _intervals(tracks: np.ndarray) -> list[float]:
    times = np.flatnonzero(tracks[:, :2].any(axis=1)) / CONFIG.audio.frames_per_sec
    return np.diff(times).tolist()


def _f1(predicted: int, target: int, matched: int) -> tuple[float, float, float]:
    precision = matched / predicted if predicted else float(target == 0)
    recall = matched / target if target else float(predicted == 0)
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def generation_score(items: list[tuple[str, np.ndarray, np.ndarray, int]]) -> float:
    """以 Tap 与持续音起点的微平均 F1 均值作为正式选模指标。"""
    if not items:
        return 0.0
    scores = []
    for channel in (TAP_COUNT, HOLD_START_COUNT):
        predicted = sum(int(value[:, channel].sum()) for _, _, value, _ in items)
        target = sum(int(value[:, channel].sum()) for _, value, _, _ in items)
        matched = sum(int(np.minimum(value[:, channel], prediction[:, channel]).sum()) for _, value, prediction, _ in items)
        scores.append(_f1(predicted, target, matched)[2])
    return mean(scores)


def _match_lines(predicted: np.ndarray, target: np.ndarray) -> list[str]:
    tap_pred, tap_target = int(predicted[:, 0].sum()), int(target[:, 0].sum())
    long_pred, long_target = int(predicted[:, 1].sum()), int(target[:, 1].sum())
    tap = _f1(tap_pred, tap_target, int(np.minimum(predicted[:, 0], target[:, 0]).sum()))
    long = _f1(long_pred, long_target, int(np.minimum(predicted[:, 1], target[:, 1]).sum()))
    duration_mask = np.stack((target[:, HOLD_START_COUNT] >= 1, target[:, HOLD_START_COUNT] >= 2), axis=1)
    duration_error = np.abs(predicted[:, 2:] - target[:, 2:])[duration_mask]
    duration = float(duration_error.mean()) if duration_error.size else 0.0
    return [
        f"    Tap 时间点匹配：精确率={tap[0] * 100:.1f}% 召回率={tap[1] * 100:.1f}% F1={tap[2] * 100:.1f}%。",
        f"    持续音起点匹配：精确率={long[0] * 100:.1f}% 召回率={long[1] * 100:.1f}% F1={long[2] * 100:.1f}%。",
        f"    真实持续音时长 MAE：{duration:.1f} 帧（{duration / CONFIG.audio.frames_per_sec:.4f} 秒）。",
    ]


def format_generation_comparison(items: list[tuple[str, np.ndarray, np.ndarray, int]]) -> list[str]:
    """输出正式后处理结果相对原谱及 Master 总体的可读比较。"""
    if not items:
        return ["  没有可比较的整曲生成结果。"]
    lines = ["正式整曲生成与原谱对比（持续音合并 Hold/Touch Hold/Slide）："]
    generated_counts, all_intervals = [], []
    for title, target, predicted, dropped in items:
        expected, actual = _counts(target), _counts(predicted)
        intervals = _intervals(predicted)
        generated_counts.append(actual)
        all_intervals.extend(intervals)
        lines.extend([
            f"  {title}：Tap={actual['tap']}/{expected['tap']} ({_ratio(actual['tap'], expected['tap'])})，持续音={actual['long']}/{expected['long']} ({_ratio(actual['long'], expected['long'])})，总音符={actual['total']}/{expected['total']} ({_ratio(actual['total'], expected['total'])})，过滤 Tap={dropped}。",
            *_match_lines(predicted, target),
        ])
    lines.append(f"Master 总体基准比较（{len(items)} 首生成谱的每首平均；基准为 1588 首 Master）：")
    for key, label in (("tap", "Tap/Touch"), ("long", "持续音"), ("total", "总音符")):
        values = [counts[key] for counts in generated_counts]
        average, middle = mean(values), median(values)
        lines.append(f"  {label}：平均={average:.1f}，为 Master 平均 {MASTER_MEAN[key]:.1f} 的 {_ratio(average, MASTER_MEAN[key])}；中位数={middle:.1f}，为 Master 参考中位数 {MASTER_MEDIAN[key]:.1f} 的 {_ratio(middle, MASTER_MEDIAN[key])}。")
    if all_intervals:
        average, middle = mean(all_intervals), median(all_intervals)
        lines.append(f"  相邻音符间隔：平均={average:.4f} 秒，为 Master 平均 {MASTER_MEAN['interval']:.4f} 秒的 {_ratio(average, MASTER_MEAN['interval'])}；中位数={middle:.4f} 秒，为 Master 中位数 {MASTER_MEDIAN['interval']:.4f} 秒的 {_ratio(middle, MASTER_MEDIAN['interval'])}。")
    else:
        lines.append("  相邻音符间隔：生成谱没有足够的不同时间点，无法比较。")
    return lines


def _self_check() -> None:
    target = np.array(((1, 1, 10, 0), (0, 0, 0, 0), (1, 0, 0, 0)), dtype=np.int32)
    predicted = np.array(((1, 1, 10, 0), (1, 0, 0, 0), (0, 0, 0, 0)), dtype=np.int32)
    text = "\n".join(format_generation_comparison([("test", target, predicted, 2)]))
    assert "Tap=2/2 (100.0%)" in text and "F1=50.0%" in text and "过滤 Tap=2" in text
    assert generation_score([("test", target, predicted, 2)]) == 0.75
    print("[generation-metrics] 自检通过")


if __name__ == "__main__":
    _self_check()
