"""单曲多难度、1024 帧窗口、10 帧步长的过拟合训练入口。"""

from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG, checkpoint_config
from chart import Chart
from dataset import (
    HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, OverfitWindowDataset,
    SongData, WINDOW_FRAMES, TRAIN_STRIDE, load_overfit_song, tracks_to_chart,
)
from infer import MODEL_KIND, infer_features, load_model, model_dimensions, save_inference
from maidata_parser import generate_maidata, parse_maidata
from model import NoteTimingTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVENT_TOLERANCE_FRAMES = 6
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")


def _duration_target(events: torch.Tensor) -> torch.Tensor:
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    return torch.log1p(events[..., HOLD_DURATION_1:].float()) / np.log1p(maximum)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(output: tuple[torch.Tensor, ...], batch: dict[str, torch.Tensor]):
    tap_logits, hold_logits, duration = output
    events, mask = batch["events"], batch["mask"]
    tap_target = events[..., TAP_COUNT].clamp_max(8)
    hold_target = events[..., HOLD_START_COUNT].clamp_max(2)
    tap_loss = nn.functional.cross_entropy(tap_logits.transpose(1, 2), tap_target, reduction="none")
    hold_loss = nn.functional.cross_entropy(hold_logits.transpose(1, 2), hold_target, reduction="none")
    valid = mask.float()
    tap_empty_loss = _masked_mean(tap_loss, valid * (tap_target == 0))
    tap_event_loss = _masked_mean(tap_loss, valid * (tap_target > 0))
    hold_empty_loss = _masked_mean(hold_loss, valid * (hold_target == 0))
    hold_event_loss = _masked_mean(hold_loss, valid * (hold_target > 0))
    count_loss = (tap_empty_loss + tap_event_loss + hold_empty_loss + hold_event_loss) / 4
    duration_mask = torch.stack((hold_target >= 1, hold_target >= 2), dim=-1) & mask.unsqueeze(-1)
    duration_error = nn.functional.smooth_l1_loss(duration, _duration_target(events), reduction="none")
    duration_loss = (duration_error * duration_mask).sum() / duration_mask.sum().clamp_min(1)
    return (
        count_loss + duration_loss,
        tap_empty_loss,
        tap_event_loss,
        hold_empty_loss,
        hold_event_loss,
        duration_loss,
    )


def run_epoch(model, loader, optimizer, scaler) -> tuple[float, ...]:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(6, dtype=np.float64)
    for batch in tqdm(loader, desc="训练" if training else "评估", leave=False):
        for key in ("features", "events", "mask", "difficulty"):
            batch[key] = batch[key].to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda",
        ):
            output = model(batch["features"], batch["difficulty"], batch["mask"])
            losses = compute_loss(output, batch)
            loss = losses[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        totals += [value.item() for value in losses]
    average = totals / max(len(loader), 1)
    return tuple(float(value) for value in average)


def _song_info(song: SongData) -> dict:
    return {
        "title": song.title,
        "music_id": song.music_id,
        "music_data_digest": song.music_data_digest,
        "levels": [(level.level_idx, level.difficulty) for level in song.levels],
    }


def _checkpoint(
    model, optimizer, scheduler, scaler, epoch: int, best_f1: float,
    song: SongData, metrics: dict,
) -> dict:
    return {
        "checkpoint_version": 1,
        "model_kind": MODEL_KIND,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": best_f1,
        "dims": model_dimensions(),
        "config": checkpoint_config(),
        "window_frames": WINDOW_FRAMES,
        "train_stride": TRAIN_STRIDE,
        "song": _song_info(song),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "metrics": metrics,
    }


def _validate_training_checkpoint(state: dict, song: SongData) -> None:
    if (
        state.get("checkpoint_version") != 1
        or state.get("model_kind") != MODEL_KIND
        or state.get("dims") != model_dimensions()
        or state.get("config") != checkpoint_config()
        or state.get("window_frames") != WINDOW_FRAMES
        or state.get("train_stride") != TRAIN_STRIDE
        or state.get("song") != _song_info(song)
    ):
        raise ValueError("训练检查点的架构、配置或歌曲数据与当前训练不一致")
def _restore_training(state, model, optimizer, scheduler, scaler) -> tuple[int, float]:
    model.load_state_dict(state["model_state_dict"])
    if "optimizer_state_dict" not in state:
        generation = state.get("metrics", {}).get("generation", {})
        best_f1 = float(generation.get("total", {}).get("f1", -1.0))
        print("检测到旧检查点：仅恢复模型参数，优化器和学习率从当前配置重新开始", flush=True)
        return int(state["epoch"]) + 1, best_f1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["epoch"]) + 1, float(state["best_f1"])


def _reached_perfect_f1(metrics: dict) -> bool:
    return math.isclose(float(metrics["total"]["f1"]), 1.0, abs_tol=1e-12)


def _events(tracks: np.ndarray, column: int) -> list[tuple[int, int]]:
    result = []
    for frame in np.flatnonzero(tracks[:, column]):
        count = int(tracks[frame, column])
        for slot in range(count):
            duration = int(tracks[frame, HOLD_DURATION_1 + slot]) if column == HOLD_START_COUNT else 0
            result.append((int(frame), duration))
    return result


def _match_events(
    target: list[tuple[int, int]], predicted: list[tuple[int, int]],
    tolerance: int = EVENT_TOLERANCE_FRAMES,
) -> tuple[int, int, int, list[int]]:
    target = sorted(target)
    predicted = sorted(predicted)
    target_index = predicted_index = matches = 0
    duration_errors: list[int] = []
    while target_index < len(target) and predicted_index < len(predicted):
        target_frame, target_duration = target[target_index]
        predicted_frame, predicted_duration = predicted[predicted_index]
        if predicted_frame < target_frame - tolerance:
            predicted_index += 1
        elif target_frame < predicted_frame - tolerance:
            target_index += 1
        else:
            matches += 1
            if target_duration:
                duration_errors.append(abs(target_duration - predicted_duration))
            target_index += 1
            predicted_index += 1
    return matches, len(predicted) - matches, len(target) - matches, duration_errors


def _scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int]:
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_tracks(target: np.ndarray, predicted: np.ndarray) -> dict:
    tap = _match_events(_events(target, TAP_COUNT), _events(predicted, TAP_COUNT))
    hold = _match_events(_events(target, HOLD_START_COUNT), _events(predicted, HOLD_START_COUNT))
    total = tuple(tap[index] + hold[index] for index in range(3))
    duration_errors = hold[3]
    return {
        "tap": _scores(*tap[:3]),
        "hold": _scores(*hold[:3]),
        "total": _scores(*total),
        "duration_error_frames": duration_errors,
    }


def _merge_evaluations(evaluations: list[dict]) -> dict:
    result = {}
    for kind in ("tap", "hold", "total"):
        totals = [sum(int(item[kind][key]) for item in evaluations) for key in ("tp", "fp", "fn")]
        result[kind] = _scores(*totals)
    errors = [value for item in evaluations for value in item["duration_error_frames"]]
    result["duration_mae_ms"] = (
        float(np.mean(errors) / CONFIG.audio.frames_per_sec * 1000) if errors else None
    )
    return result


@torch.no_grad()
def generate_song(model: NoteTimingTransformer, song: SongData, output_root: Path | None = None) -> dict:
    evaluations = []
    generated = Chart(title=song.title, artist="generated")
    for level in song.levels:
        predicted, _ = infer_features(model, song.features, level.difficulty, DEVICE)
        evaluations.append(evaluate_tracks(level.tracks, predicted))
        if output_root is not None:
            generated.all_levels[level.level_idx] = tracks_to_chart(
                predicted, level.level_idx, song.title, level.difficulty,
            ).all_levels[level.level_idx]
    if output_root is not None:
        save_inference(song.audio_path, generate_maidata(generated), output_root)
    return _merge_evaluations(evaluations)


def _format_event_metrics(label: str, metrics: dict) -> str:
    return (
        f"{label}: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
        f"Precision={metrics['precision']:.2%} Recall={metrics['recall']:.2%} F1={metrics['f1']:.2%}"
    )


def format_epoch_log(epoch: int, updates: int, losses: tuple[float, ...], metrics: dict, lr: float) -> str:
    duration = metrics["duration_mae_ms"]
    duration_text = "无可计算样本" if duration is None else f"{duration:.2f}ms"
    return "\n".join((
        f"epoch={epoch} 本轮参数更新={updates} lr={lr:.3e}",
        f"优化目标={losses[0]:.6f}",
        f"  Tap 空白帧交叉熵={losses[1]:.6f}",
        f"  Tap 事件帧交叉熵={losses[2]:.6f}",
        f"  Hold 空白帧交叉熵={losses[3]:.6f}",
        f"  Hold 事件帧交叉熵={losses[4]:.6f}",
        f"  时长归一化优化损失={losses[5]:.6f}",
        f"整曲正式推理（起点容差=±{EVENT_TOLERANCE_FRAMES}帧/{EVENT_TOLERANCE_FRAMES / CONFIG.audio.frames_per_sec * 1000:g}ms）：",
        _format_event_metrics("  Tap", metrics["tap"]),
        _format_event_metrics("  Hold", metrics["hold"]),
        _format_event_metrics("  全部事件", metrics["total"]),
        f"  匹配 Hold 时长 MAE={duration_text}",
    ))


def _self_check() -> None:
    device = torch.device("cpu")
    batch = {
        "events": torch.zeros(1, 8, 4, dtype=torch.long, device=device),
        "mask": torch.ones(1, 8, dtype=torch.bool, device=device),
    }
    output = (
        torch.zeros(1, 8, 9, device=device),
        torch.zeros(1, 8, 3, device=device),
        torch.full((1, 8, 2), 0.5, device=device),
    )
    empty_loss = compute_loss(output, batch)
    assert all(value.isfinite() for value in empty_loss)
    batch["events"][0, 0] = torch.tensor((1, 1, 20, 0), device=device)
    missed_event_loss = compute_loss(output, batch)
    assert missed_event_loss[2] > 1 and missed_event_loss[4] > 1 and missed_event_loss[5] > 0

    correct = (
        torch.full((1, 8, 9), -10.0, device=device),
        torch.full((1, 8, 3), -10.0, device=device),
        _duration_target(batch["events"]),
    )
    correct[0][..., 0] = 10
    correct[0][0, 0, 0] = -10
    correct[0][0, 0, 1] = 10
    correct[1][..., 0] = 10
    correct[1][0, 0, 0] = -10
    correct[1][0, 0, 1] = 10
    assert compute_loss(correct, batch)[0] < missed_event_loss[0]

    target = np.zeros((30, 4), dtype=np.int32)
    predicted = np.zeros_like(target)
    target[10] = (2, 1, 20, 0)
    predicted[4] = (1, 0, 0, 0)
    predicted[10] = (1, 0, 0, 0)
    predicted[16] = (0, 1, 24, 0)
    metrics = evaluate_tracks(target, predicted)
    assert metrics["tap"] == _scores(2, 0, 0)
    assert metrics["hold"] == _scores(1, 0, 0)
    assert metrics["duration_error_frames"] == [4]
    predicted[:] = 0
    predicted[17, TAP_COUNT] = 1
    assert evaluate_tracks(target, predicted)["tap"] == _scores(0, 1, 2)
    assert evaluate_tracks(target, predicted)["hold"] == _scores(0, 0, 1)
    merged = _merge_evaluations([evaluate_tracks(target, np.zeros_like(target))])
    assert merged["total"]["recall"] == 0 and merged["total"]["f1"] == 0
    assert _reached_perfect_f1({"total": {"f1": 1.0}})
    assert not _reached_perfect_f1({"total": {"f1": 0.999999}})
    restore_model = nn.Linear(2, 1)
    restore_optimizer = torch.optim.AdamW(restore_model.parameters())
    restore_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restore_optimizer, 2)
    restore_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restore_state = {
        "epoch": 3,
        "model_state_dict": restore_model.state_dict(),
        "optimizer_state_dict": restore_optimizer.state_dict(),
        "scheduler_state_dict": restore_scheduler.state_dict(),
        "scaler_state_dict": restore_scaler.state_dict(),
        "best_f1": 0.75,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": None,
        "numpy_rng_state": np.random.get_state(),
    }
    with torch.no_grad():
        restore_model.weight.zero_()
    start_epoch, best_f1 = _restore_training(
        restore_state, restore_model, restore_optimizer, restore_scheduler, restore_scaler,
    )
    assert start_epoch == 4 and best_f1 == 0.75
    assert torch.equal(restore_model.weight, restore_state["model_state_dict"]["weight"])
    combined = Chart(title="test", artist="generated")
    for level_idx, difficulty in ((2, 6.0), (5, 13.7)):
        level_tracks = np.zeros((2, 4), dtype=np.int32)
        level_tracks[1, TAP_COUNT] = 1
        combined.all_levels[level_idx] = tracks_to_chart(
            level_tracks, level_idx, "test", difficulty,
        ).all_levels[level_idx]
    restored = parse_maidata(generate_maidata(combined))
    assert restored.all_levels[2].level_query == 6.0
    assert restored.all_levels[5].level_query == 13.7
    assert "TP=" in format_epoch_log(1, 10, tuple(float(x) for x in missed_event_loss), merged, 1e-4)
    print("[train] 自检通过")


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    song = load_overfit_song()
    dataset = OverfitWindowDataset(song)
    options = {
        "batch_size": CONFIG.training.batch_size,
        "shuffle": True,
        "num_workers": CONFIG.training.num_workers,
        "pin_memory": CONFIG.training.pin_memory,
    }
    if CONFIG.training.num_workers:
        options["prefetch_factor"] = CONFIG.training.prefetch_factor
    loader = DataLoader(dataset, **options)
    model = NoteTimingTransformer(model_dimensions()).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.training.learning_rate,
        weight_decay=CONFIG.training.weight_decay, fused=DEVICE.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.training.lr_t_max, eta_min=CONFIG.training.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"设备={DEVICE} 歌曲={song.title} 难度={[(x.level_idx, x.difficulty) for x in song.levels]} "
        f"窗口={WINDOW_FRAMES} 步长={TRAIN_STRIDE} 样本={len(dataset)} 参数={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    start_epoch, best_f1 = 1, -1.0
    best_path = CONFIG.paths.checkpoint_dir / "best.pt"
    newest_path = CONFIG.paths.checkpoint_dir / "newest.pt"
    if newest_path.is_file():
        state = torch.load(newest_path, map_location=DEVICE, weights_only=False)
        _validate_training_checkpoint(state, song)
        start_epoch, best_f1 = _restore_training(state, model, optimizer, scheduler, scaler)
        print(
            f"从 {newest_path} 恢复：已完成 epoch={state['epoch']}，"
            f"从 epoch={start_epoch} 继续，历史最佳整曲事件 F1={best_f1:.2%}",
            flush=True,
        )
    for epoch in range(start_epoch, CONFIG.training.num_epochs + 1):
        losses = run_epoch(model, loader, optimizer, scaler)
        scheduler.step()
        metrics = generate_song(model, song)
        values = {
            "optimization": {
                "loss": losses[0],
                "tap_empty_loss": losses[1],
                "tap_event_loss": losses[2],
                "hold_empty_loss": losses[3],
                "hold_event_loss": losses[4],
                "duration_loss": losses[5],
            },
            "generation": metrics,
        }
        improved = metrics["total"]["f1"] > best_f1
        if improved:
            best_f1 = metrics["total"]["f1"]
        state = _checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1, song, values)
        torch.save(state, newest_path)
        if improved:
            torch.save(state, best_path)
        print(format_epoch_log(epoch, len(loader), losses, metrics, scheduler.get_last_lr()[0]), flush=True)
        if _reached_perfect_f1(metrics):
            print(f"整曲事件 F1 已达到 100%，在 epoch={epoch} 提前停止", flush=True)
            break
    if not best_path.is_file():
        raise ValueError(f"没有可用于正式推理的最佳检查点: {best_path}")
    print(f"训练结束，使用正式推理流程加载 {best_path}", flush=True)
    inference_model = load_model(best_path, DEVICE)
    metrics = generate_song(inference_model, song, CONFIG.paths.overfit_output_dir / song.title)
    print(
        f"过拟合生成完成，最佳整曲事件 F1={metrics['total']['f1']:.2%} "
        f"输出={CONFIG.paths.overfit_output_dir / song.title}"
    )


if __name__ == "__main__":
    main()
