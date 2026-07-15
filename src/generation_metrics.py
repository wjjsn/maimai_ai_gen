"""评估正式后处理后的简化事件时间轴。"""

from dataclasses import dataclass
from statistics import mean, median

import numpy as np

from config import CONFIG
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT


MATCH_TOLERANCE_FRAMES = round(0.01 * CONFIG.audio.frames_per_sec)
DURATION_TOLERANCE_FRAMES = round(0.05 * CONFIG.audio.frames_per_sec)


@dataclass(frozen=True)
class EvaluationItem:
    label: str
    level_query: float
    target: np.ndarray
    predicted: np.ndarray
    dropped: int


def _events(tracks: np.ndarray, channel: int) -> list[int]:
    return [frame for frame, count in enumerate(tracks[:, channel]) for _ in range(int(count))]


def _long_events(tracks: np.ndarray) -> list[tuple[int, int]]:
    return [
        (frame, int(values[HOLD_DURATION_1 + slot]))
        for frame, values in enumerate(tracks)
        for slot in range(min(2, int(values[HOLD_START_COUNT])))
    ]


def _match(left: list[int], right: list[int], tolerance: int) -> list[tuple[int, int]]:
    """按时间顺序做最大数量的一对一容差匹配。"""
    pairs = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j] - tolerance:
            i += 1
        elif right[j] < left[i] - tolerance:
            j += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    return pairs


def _prf(predicted: int, target: int, matched: int) -> dict[str, float]:
    precision = matched / predicted if predicted else float(target == 0)
    recall = matched / target if target else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _channel_metrics(items: list[EvaluationItem], channel: int, tolerance: int) -> dict[str, float]:
    predicted = target = matched = 0
    for item in items:
        prediction_events = _events(item.predicted, channel)
        target_events = _events(item.target, channel)
        predicted += len(prediction_events)
        target += len(target_events)
        matched += len(_match(prediction_events, target_events, tolerance))
    return _prf(predicted, target, matched)


def _long_accuracy(items: list[EvaluationItem]) -> tuple[float | None, float | None, list[int]]:
    """统计预测为持续的类型准确率，以及匹配持续音中的时长准确率。"""
    predicted_count = matched_count = duration_correct = 0
    duration_errors = []
    for item in items:
        predicted = _long_events(item.predicted)
        target = _long_events(item.target)
        predicted_count += len(predicted)
        for predicted_index, target_index in _match(
            [frame for frame, _ in predicted], [frame for frame, _ in target], MATCH_TOLERANCE_FRAMES
        ):
            error = abs(predicted[predicted_index][1] - target[target_index][1])
            matched_count += 1
            duration_errors.append(error)
            duration_correct += error <= DURATION_TOLERANCE_FRAMES
    return (
        matched_count / predicted_count if predicted_count else None,
        duration_correct / matched_count if matched_count else None,
        duration_errors,
    )


def _counts(tracks: np.ndarray) -> dict[str, int]:
    tap = int(tracks[:, TAP_COUNT].sum())
    long = int(tracks[:, HOLD_START_COUNT].sum())
    return {"tap": tap, "long": long, "total": tap + long}


def _intervals(tracks: np.ndarray) -> list[float]:
    times = np.flatnonzero(tracks[:, :2].any(axis=1)) / CONFIG.audio.frames_per_sec
    return np.diff(times).tolist()


def _average_middle(values: list[float]) -> tuple[float, float]:
    return (float(mean(values)), float(median(values))) if values else (0.0, 0.0)


def build_level_reference(items: list[tuple[float, np.ndarray]]) -> dict[float, dict]:
    """按 Diving-Fish 精确定数汇总完整数据集基准。"""
    groups: dict[float, list[np.ndarray]] = {}
    for level_query, tracks in items:
        groups.setdefault(level_query, []).append(tracks)
    reference = {}
    for level_query, tracks_group in groups.items():
        counts = [_counts(tracks) for tracks in tracks_group]
        values = {"charts": len(tracks_group)}
        for key in ("tap", "long", "total"):
            values[key] = _average_middle([value[key] for value in counts])
        values["interval"] = _average_middle([
            value for tracks in tracks_group for value in _intervals(tracks)
        ])
        reference[level_query] = values
    return reference


def _level_metrics(items: list[EvaluationItem], reference: dict[float, dict] | None) -> list[dict]:
    groups: dict[float, list[EvaluationItem]] = {}
    for item in items:
        groups.setdefault(item.level_query, []).append(item)
    rows = []
    for level_query, group in sorted(groups.items()):
        generated = [_counts(item.predicted) for item in group]
        targets = [_counts(item.target) for item in group]
        baseline = reference.get(level_query) if reference is not None else None
        row = {
            "level_query": level_query,
            "charts": len(group),
            "reference_charts": baseline["charts"] if baseline is not None else len(group),
        }
        for key in ("tap", "long", "total"):
            row[f"generated_{key}"] = _average_middle([value[key] for value in generated])
            row[f"target_{key}"] = baseline[key] if baseline is not None else _average_middle([value[key] for value in targets])
        row["generated_interval"] = _average_middle([
            value for item in group for value in _intervals(item.predicted)
        ])
        row["target_interval"] = baseline["interval"] if baseline is not None else _average_middle([
            value for item in group for value in _intervals(item.target)
        ])
        tap = _channel_metrics(group, TAP_COUNT, MATCH_TOLERANCE_FRAMES)
        long = _channel_metrics(group, HOLD_START_COUNT, MATCH_TOLERANCE_FRAMES)
        row["score"] = mean((tap["f1"], long["f1"]))
        row["long_type_accuracy"], row["long_duration_accuracy"], _ = _long_accuracy(group)
        rows.append(row)
    return rows


def evaluate_generation(items: list[EvaluationItem], reference: dict[float, dict] | None = None) -> dict:
    if not items:
        return {
            "score": 0.0, "strict_score": 0.0,
            "tap": _prf(0, 0, 0), "long": _prf(0, 0, 0),
            "duration_mae_frames": 0.0, "duration_median_frames": 0.0,
            "long_type_accuracy": None, "long_duration_accuracy": None,
            "matched_durations": 0, "generated_count": 0, "target_count": 0,
            "count_error": 0.0, "dropped": 0, "dropped_rate": 0.0,
            "empty_rate": 0.0, "levels": [],
        }

    tap = _channel_metrics(items, TAP_COUNT, MATCH_TOLERANCE_FRAMES)
    long = _channel_metrics(items, HOLD_START_COUNT, MATCH_TOLERANCE_FRAMES)
    strict_tap = _channel_metrics(items, TAP_COUNT, 0)
    strict_long = _channel_metrics(items, HOLD_START_COUNT, 0)
    long_type_accuracy, long_duration_accuracy, duration_errors = _long_accuracy(items)

    generated_count = sum(_counts(item.predicted)["total"] for item in items)
    target_count = sum(_counts(item.target)["total"] for item in items)
    dropped = sum(item.dropped for item in items)
    raw_taps = sum(_counts(item.predicted)["tap"] for item in items) + dropped
    empty = sum(_counts(item.predicted)["total"] == 0 for item in items)
    return {
        "score": mean((tap["f1"], long["f1"])),
        "strict_score": mean((strict_tap["f1"], strict_long["f1"])),
        "tap": tap,
        "long": long,
        "duration_mae_frames": float(mean(duration_errors)) if duration_errors else 0.0,
        "duration_median_frames": float(median(duration_errors)) if duration_errors else 0.0,
        "long_type_accuracy": long_type_accuracy,
        "long_duration_accuracy": long_duration_accuracy,
        "matched_durations": len(duration_errors),
        "generated_count": generated_count,
        "target_count": target_count,
        "count_error": (generated_count - target_count) / target_count if target_count else 0.0,
        "dropped": dropped,
        "dropped_rate": dropped / raw_taps if raw_taps else 0.0,
        "empty_rate": empty / len(items),
        "levels": _level_metrics(items, reference),
    }


def _self_check() -> None:
    target = np.array(((1, 1, 10, 0), (0, 0, 0, 0), (1, 0, 0, 0)), dtype=np.int32)
    predicted = np.array(((0, 0, 0, 0), (1, 1, 12, 0), (1, 0, 0, 0)), dtype=np.int32)
    item = EvaluationItem("test", 13.7, target, predicted, 2)
    metrics = evaluate_generation([item])
    assert metrics["score"] == 1.0 and metrics["strict_score"] == 0.25
    assert metrics["duration_mae_frames"] == 2 and metrics["matched_durations"] == 1
    assert metrics["long_type_accuracy"] == 1.0 and metrics["long_duration_accuracy"] == 1.0
    assert metrics["dropped_rate"] == 0.5 and metrics["levels"][0]["level_query"] == 13.7

    duration_target = np.array(((0, 1, 100, 0), (0, 1, 100, 0)), dtype=np.int32)
    duration_predicted = np.array(((0, 1, 110, 0), (0, 1, 111, 0)), dtype=np.int32)
    duration_metrics = evaluate_generation([EvaluationItem("duration", 13.7, duration_target, duration_predicted, 0)])
    assert duration_metrics["long_type_accuracy"] == 1.0 and duration_metrics["long_duration_accuracy"] == 0.5

    no_long_metrics = evaluate_generation([EvaluationItem("no-long", 13.7, np.zeros((1, 4), dtype=np.int32), np.zeros((1, 4), dtype=np.int32), 0)])
    assert no_long_metrics["long_type_accuracy"] is None and no_long_metrics["long_duration_accuracy"] is None

    duplicate = np.array(((2, 0, 0, 0),), dtype=np.int32)
    single = np.array(((1, 0, 0, 0),), dtype=np.int32)
    score = evaluate_generation([EvaluationItem("duplicate", 13.1, single, duplicate, 0)])["tap"]
    assert score == {"precision": 0.5, "recall": 1.0, "f1": 2 / 3}
    print("[generation-metrics] 自检通过")


if __name__ == "__main__":
    _self_check()
