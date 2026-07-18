"""全量多歌曲、多难度逐帧事件模型训练入口。"""

import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import CONFIG, checkpoint_config, max_hold_duration_frames
from dataset import (
    HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, DatasetIndex, SongRecord,
    build_dataset_index, iter_loaded_window_batches, iter_window_batches,
    load_song_data, split_songs, _shift_window,
)
from infer import LABEL_PROTOCOL, MODEL_KIND, infer_features, model_dimensions
from model import NoteTimingTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOLERANCES = (0, 3, 6, 10)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")


def _duration_target(events: torch.Tensor) -> torch.Tensor:
    maximum = max_hold_duration_frames()
    return torch.log1p(events[..., HOLD_DURATION_1:].float()) / np.log1p(maximum)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _duration_supervision_mask(
    events: torch.Tensor, mask: torch.Tensor, duration_anchor: torch.Tensor,
) -> torch.Tensor:
    hold_count = events[..., HOLD_START_COUNT]
    durations = events[..., HOLD_DURATION_1:]
    slots = torch.stack((hold_count >= 1, hold_count >= 2), dim=-1)
    positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1, 1)
    valid_end = torch.where(
        mask, torch.arange(1, mask.shape[1] + 1, device=mask.device), 0,
    ).amax(dim=1, keepdim=True).unsqueeze(-1)
    return (
        slots
        & duration_anchor.view(-1, 1, 1)
        & (positions == 0)
        & mask.unsqueeze(-1)
        & (durations > 0)
        & (durations <= max_hold_duration_frames())
        & (positions + durations <= valid_end)
    )


def _event_heatmap(target: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    centers = (target > 0).float()
    heatmap = centers.clone()
    for distance in range(1, radius + 1):
        value = 1 - distance / (radius + 1)
        heatmap[:, distance:] = torch.maximum(heatmap[:, distance:], centers[:, :-distance] * value)
        heatmap[:, :-distance] = torch.maximum(heatmap[:, :-distance], centers[:, distance:] * value)
    return heatmap * mask


def _event_margin(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., 1:].amax(dim=-1) - logits[..., 0]


def _presence_loss(logits: torch.Tensor, heatmap: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(
        _event_margin(logits), heatmap, reduction="none",
    )
    event_region = valid * (heatmap > 0)
    background = valid * (heatmap == 0)
    return (_masked_mean(loss, event_region) + _masked_mean(loss, background)) / 2


def compute_loss(output: tuple[torch.Tensor, ...], batch: dict[str, torch.Tensor]):
    tap_logits, hold_logits, duration = output
    events, mask = batch["events"], batch["mask"]
    tap_target = events[..., TAP_COUNT].clamp_max(2)
    hold_target = events[..., HOLD_START_COUNT].clamp_max(2)
    valid = mask.float()
    radius = CONFIG.inference.short_min_gap_frames
    tap_presence_loss = _presence_loss(
        tap_logits, _event_heatmap(tap_target, valid, radius), valid,
    )
    hold_presence_loss = _presence_loss(
        hold_logits, _event_heatmap(hold_target, valid, radius), valid,
    )
    tap_count_loss = _masked_mean(nn.functional.cross_entropy(
        tap_logits[..., 1:].transpose(1, 2), (tap_target - 1).clamp_min(0), reduction="none",
    ), valid * (tap_target > 0))
    hold_count_loss = _masked_mean(nn.functional.cross_entropy(
        hold_logits[..., 1:].transpose(1, 2), (hold_target - 1).clamp_min(0), reduction="none",
    ), valid * (hold_target > 0))
    count_loss = (tap_presence_loss + hold_presence_loss + tap_count_loss + hold_count_loss) / 4
    duration_mask = _duration_supervision_mask(events, mask, batch["duration_anchor"])
    duration_error = nn.functional.smooth_l1_loss(duration, _duration_target(events), reduction="none")
    duration_loss = (duration_error * duration_mask).sum() / duration_mask.sum().clamp_min(1)
    return (
        count_loss + duration_loss,
        tap_presence_loss,
        tap_count_loss,
        hold_presence_loss,
        hold_count_loss,
        duration_loss,
    )


def build_scheduler(optimizer, total_epochs: int):
    warmup_epochs = round(total_epochs * CONFIG.training.warmup_ratio)
    minimum_ratio = CONFIG.training.min_learning_rate / CONFIG.training.learning_rate

    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, 1 / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * min(progress, 1))) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_scheduler_state(scheduler, optimizer, state: dict) -> None:
    scheduler.load_state_dict(state)
    learning_rates = scheduler.get_last_lr()
    if len(learning_rates) != len(optimizer.param_groups):
        raise ValueError("学习率调度器与优化器参数组数量不一致")
    for group, learning_rate in zip(optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate


def run_epoch(
    model, batches, optimizer, scheduler, scaler, description: str, total: int | None = None,
) -> tuple[tuple[float, ...], int]:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(6, dtype=np.float64)
    count = 0
    for batch in tqdm(batches, desc=description, total=total, leave=False):
        for key in ("features", "events", "mask", "duration_anchor", "difficulty"):
            batch[key] = batch[key].to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda",
        ):
            losses = compute_loss(model(batch["features"], batch["difficulty"], batch["mask"]), batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses[0]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        totals += [value.item() for value in losses]
        count += 1
    if not count:
        raise RuntimeError(f"{description}没有生成任何批次")
    return tuple(float(value) for value in totals / count), count


def _events(tracks: np.ndarray, column: int) -> list[tuple[int, int]]:
    result = []
    for frame in np.flatnonzero(tracks[:, column]):
        for slot in range(int(tracks[frame, column])):
            duration = int(tracks[frame, HOLD_DURATION_1 + slot]) if column == HOLD_START_COUNT else 0
            result.append((int(frame), duration))
    return result


def _match_events(target, predicted, tolerance: int, target_length: int | None = None):
    target, predicted = sorted(target), sorted(predicted)
    target_index = predicted_index = matches = 0
    start_errors: list[int] = []
    duration_errors: list[int] = []
    duration_samples: list[tuple[int, int]] = []
    while target_index < len(target) and predicted_index < len(predicted):
        target_frame, target_duration = target[target_index]
        predicted_frame, predicted_duration = predicted[predicted_index]
        if predicted_frame < target_frame - tolerance:
            predicted_index += 1
        elif target_frame < predicted_frame - tolerance:
            target_index += 1
        else:
            matches += 1
            start_errors.append(abs(target_frame - predicted_frame))
            if target_duration:
                error = abs(target_duration - predicted_duration)
                endpoint_visible = target_length is None or target_frame + target_duration <= target_length
                if endpoint_visible:
                    duration_samples.append((target_duration, error))
                    if target_duration <= max_hold_duration_frames():
                        duration_errors.append(error)
            target_index += 1
            predicted_index += 1
    return (
        matches, len(predicted) - matches, len(target) - matches,
        start_errors, duration_errors, duration_samples,
    )


def _scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def evaluate_tracks(target: np.ndarray, predicted: np.ndarray, tolerance: int) -> dict:
    tap = _match_events(_events(target, TAP_COUNT), _events(predicted, TAP_COUNT), tolerance)
    hold = _match_events(
        _events(target, HOLD_START_COUNT), _events(predicted, HOLD_START_COUNT),
        tolerance, len(target),
    )
    total = tuple(tap[index] + hold[index] for index in range(3))
    return {
        "tap": _scores(*tap[:3]),
        "hold": _scores(*hold[:3]),
        "total": _scores(*total),
        "start_errors": tap[3] + hold[3],
        "duration_errors": hold[4],
        "duration_samples": hold[5],
        "tap_confusion": np.bincount(
            target[:, TAP_COUNT].clip(0, 2) * 3 + predicted[:, TAP_COUNT].clip(0, 2), minlength=9,
        ).reshape(3, 3).tolist(),
        "hold_confusion": np.bincount(
            target[:, HOLD_START_COUNT].clip(0, 2) * 3 + predicted[:, HOLD_START_COUNT].clip(0, 2), minlength=9,
        ).reshape(3, 3).tolist(),
    }


def _percentiles(values: list[int], scale: float = 1.0) -> dict[str, float | None]:
    if not values:
        return {"mae": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "mae": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _duration_error_buckets(samples: list[tuple[int, int]]) -> dict[str, dict]:
    one_second = round(CONFIG.audio.frames_per_sec)
    two_point_five_seconds = round(2.5 * CONFIG.audio.frames_per_sec)
    maximum = max_hold_duration_frames()
    groups = {
        "<1s": [], "1-2.5s": [], "2.5s-window_limit": [], ">window_limit": [],
    }
    for target, error in samples:
        if target > maximum:
            name = ">window_limit"
        elif target < one_second:
            name = "<1s"
        elif target < two_point_five_seconds:
            name = "1-2.5s"
        else:
            name = "2.5s-window_limit"
        groups[name].append(error)
    scale = 1000 / CONFIG.audio.frames_per_sec
    return {
        name: {"count": len(values), **_percentiles(values, scale)}
        for name, values in groups.items()
    }


def merge_evaluations(evaluations: list[dict]) -> dict:
    result = {}
    for kind in ("tap", "hold", "total"):
        totals = [sum(int(item[kind][key]) for item in evaluations) for key in ("tp", "fp", "fn")]
        micro = _scores(*totals)
        micro["macro_f1"] = float(np.mean([item[kind]["f1"] for item in evaluations])) if evaluations else 0.0
        result[kind] = micro
    result["start_error_frames"] = _percentiles([
        value for item in evaluations for value in item["start_errors"]
    ])
    result["duration_error_ms"] = _percentiles([
        value for item in evaluations for value in item["duration_errors"]
    ], 1000 / CONFIG.audio.frames_per_sec)
    result["duration_error_buckets_ms"] = _duration_error_buckets([
        value for item in evaluations for value in item["duration_samples"]
    ])
    for name in ("tap_confusion", "hold_confusion"):
        result[name] = np.sum([item[name] for item in evaluations], axis=0).astype(int).tolist() if evaluations else [[0] * 3 for _ in range(3)]
    return result


def _combine_raw_evaluations(evaluations: list[dict]) -> dict:
    combined = {}
    for kind in ("tap", "hold", "total"):
        scores = _scores(*[
            sum(int(item[kind][key]) for item in evaluations) for key in ("tp", "fp", "fn")
        ])
        combined[kind] = scores
    combined["start_errors"] = [value for item in evaluations for value in item["start_errors"]]
    combined["duration_errors"] = [value for item in evaluations for value in item["duration_errors"]]
    combined["duration_samples"] = [value for item in evaluations for value in item["duration_samples"]]
    combined["tap_confusion"] = np.sum([item["tap_confusion"] for item in evaluations], axis=0).astype(int).tolist()
    combined["hold_confusion"] = np.sum([item["hold_confusion"] for item in evaluations], axis=0).astype(int).tolist()
    return combined


@torch.no_grad()
def evaluate_loaded_songs(model: NoteTimingTransformer, loaded_songs) -> dict:
    by_tolerance: dict[int, list[dict]] = {value: [] for value in TOLERANCES}
    by_band: dict[str, list[dict]] = {"<7": [], "7-9.5": [], "10-12.5": [], ">=13": []}
    for song, features, levels in tqdm(loaded_songs, desc="整曲验证", leave=False):
        song_evaluations: dict[int, list[dict]] = {value: [] for value in TOLERANCES}
        for level, target in levels:
            predicted, _ = infer_features(model, features, level.difficulty, DEVICE)
            for tolerance in TOLERANCES:
                evaluation = evaluate_tracks(target, predicted, tolerance)
                song_evaluations[tolerance].append(evaluation)
                if tolerance == 6:
                    band = "<7" if level.difficulty < 7 else "7-9.5" if level.difficulty < 10 else "10-12.5" if level.difficulty < 13 else ">=13"
                    by_band[band].append(evaluation)
        for tolerance in TOLERANCES:
            by_tolerance[tolerance].append(_combine_raw_evaluations(song_evaluations[tolerance]))
    return {
        "tolerances": {str(value): merge_evaluations(items) for value, items in by_tolerance.items()},
        "difficulty_bands": {name: merge_evaluations(items) for name, items in by_band.items()},
    }


def load_songs(songs: tuple[SongRecord, ...]):
    return ((song, *load_song_data(song)) for song in songs)


def evaluate_songs(model: NoteTimingTransformer, songs: tuple[SongRecord, ...]) -> dict:
    return evaluate_loaded_songs(model, load_songs(songs))


@torch.no_grad()
def validate_songs(model: NoteTimingTransformer, songs: tuple[SongRecord, ...], scaler):
    loss_totals = np.zeros(6, dtype=np.float64)
    batch_count = 0
    by_tolerance: dict[int, list[dict]] = {value: [] for value in TOLERANCES}
    by_band: dict[str, list[dict]] = {"<7": [], "7-9.5": [], "10-12.5": [], ">=13": []}
    for song in tqdm(songs, desc="验证", leave=False):
        features, levels = load_song_data(song)
        loaded = ((song, features, levels),)
        losses, batches = run_epoch(
            model, iter_loaded_window_batches(loaded), None, None, scaler, "窗口验证",
        )
        loss_totals += np.asarray(losses) * batches
        batch_count += batches
        song_evaluations: dict[int, list[dict]] = {value: [] for value in TOLERANCES}
        for level, target in levels:
            predicted, _ = infer_features(model, features, level.difficulty, DEVICE)
            for tolerance in TOLERANCES:
                evaluation = evaluate_tracks(target, predicted, tolerance)
                song_evaluations[tolerance].append(evaluation)
                if tolerance == 6:
                    band = "<7" if level.difficulty < 7 else "7-9.5" if level.difficulty < 10 else "10-12.5" if level.difficulty < 13 else ">=13"
                    by_band[band].append(evaluation)
        for tolerance in TOLERANCES:
            by_tolerance[tolerance].append(_combine_raw_evaluations(song_evaluations[tolerance]))
    generation = {
        "tolerances": {str(value): merge_evaluations(items) for value, items in by_tolerance.items()},
        "difficulty_bands": {name: merge_evaluations(items) for name, items in by_band.items()},
    }
    return tuple(float(value) for value in loss_totals / batch_count), generation


def _split_summary(index: DatasetIndex, splits: dict[str, tuple[SongRecord, ...]]) -> dict:
    return {
        "dataset_digest": index.digest,
        "music_data_digest": index.music_data_digest,
        "splits": {
            name: {
                "songs": len(songs),
                "levels": sum(len(song.levels) for song in songs),
                "ids": sorted({song.music_id for song in songs}),
            }
            for name, songs in splits.items()
        },
        "excluded": index.excluded,
        "label_normalization": index.label_normalization,
    }


def _checkpoint(
    model, optimizer, scheduler, scaler, epoch: int, global_step: int,
    best_f1: float, stale_epochs: int, split_summary: dict, metrics: dict,
) -> dict:
    return {
        "checkpoint_version": 3,
        "model_kind": MODEL_KIND,
        "label_protocol": LABEL_PROTOCOL,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": best_f1,
        "stale_epochs": stale_epochs,
        "dims": model_dimensions(),
        "config": checkpoint_config(),
        "training_config": vars(CONFIG.training),
        "augmentation_config": vars(CONFIG.augmentation),
        "split_summary": split_summary,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "metrics": metrics,
    }


def _validate_training_checkpoint(state: dict, split_summary: dict) -> bool:
    saved_config = state.get("config", {})
    expected_config = checkpoint_config()
    incompatible = (
        state.get("checkpoint_version") != 3
        or state.get("model_kind") != MODEL_KIND
        or state.get("label_protocol") != LABEL_PROTOCOL
        or state.get("dims") != model_dimensions()
        or any(saved_config.get(key) != value for key, value in expected_config.items())
    )
    if incompatible:
        raise ValueError("训练检查点的模型架构、音频特征、窗口或标签协议不兼容")
    changed = []
    if state.get("training_config") != vars(CONFIG.training):
        changed.append("训练配置")
    if state.get("augmentation_config") != vars(CONFIG.augmentation):
        changed.append("数据增强")
    if state.get("split_summary") != split_summary:
        changed.append("数据划分")
    if changed:
        print(f"警告: 检查点的{'、'.join(changed)}与当前配置不同，继续完整恢复", flush=True)
    schedule_keys = {
        "num_epochs", "learning_rate", "min_learning_rate", "warmup_ratio",
    }
    saved_training = state.get("training_config", {})
    return all(saved_training.get(key) == getattr(CONFIG.training, key) for key in schedule_keys)


def _restore_training(state, model, optimizer, scaler):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = CONFIG.training.learning_rate
        group["initial_lr"] = CONFIG.training.learning_rate
        group["weight_decay"] = CONFIG.training.weight_decay
    scaler.load_state_dict(state["scaler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    np.random.set_state(state["numpy_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])
    return (
        int(state["epoch"]) + 1, int(state["global_step"]),
        float(state["best_f1"]), int(state["stale_epochs"]),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _save_checkpoint(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _loss_dict(losses: tuple[float, ...]) -> dict[str, float]:
    return dict(zip((
        "loss", "tap_presence_loss", "tap_count_loss", "hold_presence_loss",
        "hold_count_loss", "duration_loss",
    ), losses))


def _format_epoch(record: dict) -> str:
    strict = record["validation"]["generation"]["tolerances"]["3"]["total"]
    main = record["validation"]["generation"]["tolerances"]["6"]["total"]
    duration = record["validation"]["generation"]["tolerances"]["6"]["duration_error_ms"]["mae"]
    training_generation = record["train"].get("generation")
    training_text = ""
    if training_generation is not None:
        training_strict = training_generation["tolerances"]["3"]["total"]
        training_main = training_generation["tolerances"]["6"]["total"]
        training_text = (
            f"训练F1@3={training_strict['macro_f1']:.2%} "
            f"训练F1@6={training_main['macro_f1']:.2%} "
        )
    return (
        f"epoch={record['epoch']} step={record['global_step']} lr={record['learning_rate']:.3e} "
        f"train_loss={record['train']['loss']:.6f} val_loss={record['validation']['loss']:.6f} "
        f"{training_text}"
        f"F1@3={strict['macro_f1']:.2%} 准确率@3={strict['precision']:.2%} 召回率@3={strict['recall']:.2%} "
        f"F1@6={main['macro_f1']:.2%} 准确率@6={main['precision']:.2%} 召回率@6={main['recall']:.2%} "
        f"时长MAE={duration if duration is not None else 0:.2f}ms "
        f"耗时={record['seconds']:.1f}s 吞吐={record['windows_per_sec']:.1f}窗口/s"
    )


def _self_check() -> None:
    rng_state = torch.get_rng_state()
    expected_random = torch.rand(1)
    torch.set_rng_state(rng_state.to(DEVICE).cpu())
    assert torch.equal(torch.rand(1), expected_random)
    torch.set_rng_state(rng_state)
    batch = {
        "events": torch.zeros(1, 8, 4, dtype=torch.long),
        "mask": torch.ones(1, 8, dtype=torch.bool),
        "duration_anchor": torch.ones(1, dtype=torch.bool),
    }
    output = (
        torch.zeros(1, 8, 3), torch.zeros(1, 8, 3), torch.full((1, 8, 2), 0.5),
    )
    batch["events"][0, 0] = torch.tensor((3, 1, 20, 0))
    losses = compute_loss(output, batch)
    assert all(value.isfinite() for value in losses)
    heatmap = _event_heatmap(batch["events"][..., TAP_COUNT], batch["mask"].float(), 2)
    assert torch.allclose(heatmap[0, :3], torch.tensor([1.0, 2 / 3, 1 / 3]))
    correct = (torch.zeros(1, 8, 3), torch.zeros(1, 8, 3), _duration_target(batch["events"]))
    for logits, column in ((correct[0], TAP_COUNT), (correct[1], HOLD_START_COUNT)):
        target_heatmap = _event_heatmap(batch["events"][..., column], batch["mask"].float(), 6)
        margin = torch.logit(target_heatmap.clamp(1e-4, 1 - 1e-4))
        logits[..., 0] = -margin / 2
        logits[..., 1] = margin / 2
    correct[0][0, 0, 2] = correct[0][0, 0, 1] + 10
    assert compute_loss(correct, batch)[0] < losses[0]
    visibility_events = torch.zeros(1, 8, 4, dtype=torch.long)
    visibility_mask = torch.ones(1, 8, dtype=torch.bool)
    visibility_events[0, 0] = torch.tensor((0, 1, 8, 0))
    visibility_events[0, 1] = torch.tensor((0, 1, 7, 0))
    visibility_events[0, 4] = torch.tensor((0, 1, CONFIG.training.window_frames + 1, 0))
    duration_mask = _duration_supervision_mask(
        visibility_events, visibility_mask, torch.ones(1, dtype=torch.bool),
    )
    assert duration_mask[0, 0, 0] and not duration_mask[0, 1, 0]
    assert not duration_mask[0, 4, 0]
    assert not _duration_supervision_mask(
        visibility_events, visibility_mask, torch.zeros(1, dtype=torch.bool),
    ).any()
    tail_events = torch.zeros(1, 8, 4, dtype=torch.long)
    tail_events[0, 1] = torch.tensor((0, 1, 8, 0))
    assert not _duration_supervision_mask(
        tail_events, visibility_mask, torch.ones(1, dtype=torch.bool),
    ).any()
    shifted = {
        "features": torch.zeros(8, 1),
        "events": torch.zeros(8, 4, dtype=torch.long),
        "mask": torch.ones(8, dtype=torch.bool),
    }
    shifted["events"][0] = torch.tensor((0, 1, 8, 0))
    _shift_window(shifted, 1)
    assert not _duration_supervision_mask(
        shifted["events"].unsqueeze(0), shifted["mask"].unsqueeze(0),
        torch.ones(1, dtype=torch.bool),
    ).any()
    target = np.zeros((30, 4), dtype=np.int32)
    predicted = np.zeros_like(target)
    target[10] = (2, 1, 20, 0)
    predicted[4, TAP_COUNT] = 1
    predicted[10, TAP_COUNT] = 1
    predicted[16] = (0, 1, 24, 0)
    assert evaluate_tracks(target, predicted, 6)["total"] == _scores(3, 0, 0)
    assert evaluate_tracks(target, predicted, 5)["hold"] == _scores(0, 1, 1)
    merged = merge_evaluations([evaluate_tracks(target, predicted, 6)])
    assert merged["total"]["macro_f1"] == 1.0
    maximum = max_hold_duration_frames()
    buckets = _duration_error_buckets([
        (round(CONFIG.audio.frames_per_sec) - 1, 1),
        (round(CONFIG.audio.frames_per_sec), 2),
        (round(2.5 * CONFIG.audio.frames_per_sec) - 1, 3),
        (round(2.5 * CONFIG.audio.frames_per_sec), 4),
        (maximum, 5), (maximum + 1, 6),
    ])
    assert [bucket["count"] for bucket in buckets.values()] == [1, 2, 2, 1]
    unsupported_target = np.zeros((maximum + 2, 4), dtype=np.int32)
    unsupported_prediction = np.zeros_like(unsupported_target)
    unsupported_target[1] = (0, 1, maximum + 1, 0)
    unsupported_prediction[1] = (0, 1, maximum, 0)
    unsupported = merge_evaluations([
        evaluate_tracks(unsupported_target, unsupported_prediction, 0),
    ])
    assert unsupported["duration_error_ms"]["mae"] is None
    assert unsupported["duration_error_buckets_ms"][">window_limit"]["count"] == 1
    tail_target = np.zeros((12, 4), dtype=np.int32)
    tail_prediction = np.zeros_like(tail_target)
    tail_target[10] = (0, 1, 4, 0)
    tail_prediction[10] = (0, 1, 2, 0)
    tail_metrics = merge_evaluations([
        evaluate_tracks(tail_target, tail_prediction, 0),
    ])
    assert tail_metrics["duration_error_ms"]["mae"] is None
    assert sum(
        bucket["count"] for bucket in tail_metrics["duration_error_buckets_ms"].values()
    ) == 0
    optimizer = torch.optim.AdamW(nn.Linear(2, 1).parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, 100)
    first = scheduler.get_last_lr()[0]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert 0 < first <= 1
    assert scheduler.get_last_lr()[0] <= CONFIG.training.min_learning_rate / CONFIG.training.learning_rate + 1e-6
    resume_optimizer = torch.optim.AdamW(
        nn.Linear(2, 1).parameters(), lr=CONFIG.training.learning_rate,
    )
    resume_scheduler = build_scheduler(resume_optimizer, 100)
    for _ in range(20):
        resume_optimizer.step()
        resume_scheduler.step()
    saved_scheduler_state = resume_scheduler.state_dict()
    saved_learning_rate = resume_scheduler.get_last_lr()[0]
    restored_optimizer = torch.optim.AdamW(
        nn.Linear(2, 1).parameters(), lr=CONFIG.training.learning_rate,
    )
    restored_scheduler = build_scheduler(restored_optimizer, 100)
    _load_scheduler_state(restored_scheduler, restored_optimizer, saved_scheduler_state)
    assert restored_scheduler.get_last_lr()[0] == saved_learning_rate
    assert restored_optimizer.param_groups[0]["lr"] == saved_learning_rate
    compatible_state = {
        "checkpoint_version": 3,
        "model_kind": MODEL_KIND,
        "label_protocol": LABEL_PROTOCOL,
        "dims": model_dimensions(),
        "config": checkpoint_config(),
        "training_config": vars(CONFIG.training),
        "augmentation_config": vars(CONFIG.augmentation),
        "split_summary": {"test": True},
    }
    assert _validate_training_checkpoint(compatible_state, {"test": True})
    changed_state = {**compatible_state, "training_config": {
        **compatible_state["training_config"], "early_stop_patience": 1,
    }}
    assert _validate_training_checkpoint(changed_state, {"test": True})
    incompatible_state = {**compatible_state, "config": {
        **compatible_state["config"], "window_frames": CONFIG.training.window_frames + 1,
    }}
    try:
        _validate_training_checkpoint(incompatible_state, {"test": True})
    except ValueError:
        pass
    else:
        raise AssertionError("结构不兼容的检查点必须拒绝恢复")
    print("[train] 自检通过")


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    index = build_dataset_index()
    splits = split_songs(index)
    if any(not splits[name] for name in splits):
        raise ValueError("训练、验证和测试集合都必须至少包含一首歌曲")
    summary = _split_summary(index, splits)
    CONFIG.paths.train_output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG.paths.train_output_dir / "split-summary.json", summary)
    model = NoteTimingTransformer(model_dimensions()).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.training.learning_rate,
        weight_decay=CONFIG.training.weight_decay, fused=DEVICE.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    newest_path = CONFIG.paths.checkpoint_dir / "newest-v3.pt"
    best_path = CONFIG.paths.checkpoint_dir / "best-v3.pt"
    metrics_path = CONFIG.paths.train_output_dir / "metrics.jsonl"
    start_epoch, global_step, best_f1, stale_epochs = 1, 0, -1.0, 0
    scheduler_state = None
    resume_path = CONFIG.training.resume_checkpoint
    if resume_path is None:
        metrics_path.unlink(missing_ok=True)
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"恢复检查点不存在: {resume_path}")
        state = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        schedule_compatible = _validate_training_checkpoint(state, summary)
        start_epoch, global_step, best_f1, stale_epochs = _restore_training(
            state, model, optimizer, scaler,
        )
        scheduler_state = state["scheduler_state_dict"] if schedule_compatible else None
        print(f"从 {resume_path} 恢复，下一轮 epoch={start_epoch} step={global_step}", flush=True)
    if scheduler_state is None:
        for group in optimizer.param_groups:
            group["lr"] = CONFIG.training.learning_rate
            group["initial_lr"] = CONFIG.training.learning_rate
        scheduler = build_scheduler(optimizer, CONFIG.training.num_epochs)
        if global_step:
            scheduler.step(start_epoch - 1)
    else:
        scheduler = build_scheduler(optimizer, CONFIG.training.num_epochs)
        _load_scheduler_state(scheduler, optimizer, scheduler_state)
    print(
        f"设备={DEVICE} 参数={sum(p.numel() for p in model.parameters()):,} "
        f"歌曲={len(index.songs)} 划分={{训练:{len(splits['train'])},验证:{len(splits['validation'])},测试:{len(splits['test'])}}} "
        f"学习率调度轮数={CONFIG.training.num_epochs}",
        flush=True,
    )
    print(f"标签规范化={index.label_normalization}", flush=True)
    for epoch in range(start_epoch, CONFIG.training.num_epochs + 1):
        started = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_losses, updates = run_epoch(
            model, iter_window_batches(splits["train"], epoch, True), optimizer, scheduler, scaler,
            "训练",
        )
        global_step += updates
        scheduler.step()
        training_generation = (
            evaluate_songs(model, splits["train"])
            if CONFIG.training.song_limit else None
        )
        validation_losses, generation = validate_songs(model, splits["validation"], scaler)
        seconds = time.perf_counter() - started
        main_f1 = generation["tolerances"]["6"]["total"]["macro_f1"]
        improved = main_f1 > best_f1
        if improved:
            best_f1, stale_epochs = main_f1, 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": seconds,
            "windows_per_sec": updates * CONFIG.training.batch_size / max(seconds, 1e-9),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0,
            "train": {
                **_loss_dict(train_losses),
                **({"generation": training_generation} if training_generation is not None else {}),
            },
            "validation": {**_loss_dict(validation_losses), "generation": generation},
        }
        state = _checkpoint(
            model, optimizer, scheduler, scaler, epoch, global_step,
            best_f1, stale_epochs, summary, record,
        )
        _save_checkpoint(state, newest_path)
        if improved:
            _save_checkpoint(state, best_path)
        _append_jsonl(metrics_path, record)
        print(_format_epoch(record), flush=True)
        if stale_epochs >= CONFIG.training.early_stop_patience:
            print(f"验证指标连续 {stale_epochs} 轮未提升，提前停止", flush=True)
            break
    if not best_path.is_file():
        raise ValueError("训练未生成最佳检查点")
    state = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    validation_metrics = evaluate_songs(model, splits["validation"])
    test_metrics = evaluate_songs(model, splits["test"])
    _write_json(CONFIG.paths.train_output_dir / "validation-summary.json", validation_metrics)
    _write_json(CONFIG.paths.train_output_dir / "test-summary.json", test_metrics)
    print(f"训练完成，最佳验证宏 F1@6={best_f1:.2%}，测试结果已写入 {CONFIG.paths.train_output_dir}")


if __name__ == "__main__":
    main()
