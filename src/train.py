"""四头事件 CNN 训练入口；过拟合模式只改变歌曲集合。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import MERT_HIDDEN_DIM, extract_audio_features
from config import CONFIG, checkpoint_config
from dataset import SongDataset, collate_songs, discover_songs
from infer import MODEL_KIND, events_to_frames, frames_to_maidata, predict_events, save_inference_files
from model import ChartCNN, ModelDimensions
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def split_entries(entries, seed: int, overfit_charts: int):
    order = torch.randperm(len(entries), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [entries[i] for i in order]
    if overfit_charts:
        selected = shuffled[:overfit_charts]
        return selected, selected, []
    val_count, test_count = max(1, round(len(entries) * CONFIG.training.val_ratio)), max(1, round(len(entries) * CONFIG.training.test_ratio))
    return shuffled[val_count + test_count:], shuffled[:val_count], shuffled[val_count:val_count + test_count]


def compute_loss(output, batch):
    target, mask = batch["events"], batch["mask"]
    count_active = target[..., :2].amax(dim=-1) > 0
    count_weight = 1 + count_active * (CONFIG.training.short_loss_weight - 1)
    count_loss = (nn.functional.smooth_l1_loss(output[..., :2], target[..., :2], reduction="none").mean(dim=-1) * count_weight * mask).sum() / (count_weight * mask).sum().clamp_min(1)
    duration_mask = torch.stack((target[..., HOLD_START_COUNT] >= 1, target[..., HOLD_START_COUNT] >= 2), dim=-1)
    duration_error = nn.functional.smooth_l1_loss(output[..., HOLD_DURATION_1:], target[..., HOLD_DURATION_1:], reduction="none")
    duration_loss = (duration_error * duration_mask).sum() / duration_mask.sum().clamp_min(1)
    return count_loss + duration_loss, count_loss, duration_loss


def run_epoch(model, loader, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    correct = frames = 0
    for batch in tqdm(loader, desc="训练" if training else "验证", leave=False):
        for key in ("features", "events", "mask"):
            batch[key] = batch[key].to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            output = model(batch["features"], batch["mask"])
            loss, count_loss, duration_loss = compute_loss(output, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        prediction = (output * torch.tensor((8, 2, round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec), round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)), device=DEVICE)).round()
        target = (batch["events"] * torch.tensor((8, 2, round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec), round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)), device=DEVICE)).round()
        correct += ((prediction == target).all(dim=-1) & batch["mask"]).sum().item()
        frames += batch["mask"].sum().item()
        totals += (loss.item(), count_loss.item(), duration_loss.item())
    return (*((totals / max(len(loader), 1)).tolist()), correct / max(frames, 1))


@torch.no_grad()
def generation_eval(model, entries, output_dir: Path | None = None):
    stats = {"songs": 0, "events": 0, "dropped": 0}
    for entry in tqdm(entries, desc="正式整曲推理", leave=False):
        frames, song_stats = events_to_frames(predict_events(model, extract_audio_features(entry.audio_path, DEVICE), DEVICE))
        stats["songs"] += 1
        stats["events"] += song_stats["events"]
        stats["dropped"] += song_stats["dropped"]
        if output_dir is not None:
            relative = entry.chart_path.parent.relative_to(CONFIG.paths.charts_dir)
            save_inference_files(entry.audio_path, frames_to_maidata(frames, CONFIG.training.level_idx, relative.name), output_dir / relative)
    return stats


def checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics):
    return {"checkpoint_version": 4, "model_kind": MODEL_KIND, "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(), "dims": dims, "config": checkpoint_config(), "metrics": metrics}


def _self_check() -> None:
    batch = {
        "events": torch.zeros(1, 2, 4),
        "mask": torch.ones(1, 2, dtype=torch.bool),
    }
    loss, count_loss, duration_loss = compute_loss(torch.zeros(1, 2, 4), batch)
    assert loss.isfinite() and count_loss.isfinite() and duration_loss.isfinite()
    print("[train] 自检通过")


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    entries, skipped = discover_songs(CONFIG.paths.charts_dir, CONFIG.training.level_idx)
    if skipped:
        print(f"跳过 {len(skipped)} 首含非默认等待 Slide、长按溢出或无效数据的歌曲")
    minimum = CONFIG.training.overfit_charts or 3
    if len(entries) < minimum:
        raise ValueError(f"有效歌曲只有 {len(entries)} 首，需要至少 {minimum} 首")
    train_entries, val_entries, test_entries = split_entries(entries, CONFIG.training.seed, CONFIG.training.overfit_charts)
    print(f"设备={DEVICE}，训练/验证/测试={len(train_entries)}/{len(val_entries)}/{len(test_entries)} 首")
    options = {"num_workers": CONFIG.training.num_workers, "pin_memory": CONFIG.training.pin_memory, "collate_fn": collate_songs}
    if CONFIG.training.num_workers:
        options["prefetch_factor"] = CONFIG.training.prefetch_factor
    train_loader = DataLoader(SongDataset(train_entries, CONFIG.training.level_idx, augment=True), batch_size=CONFIG.training.batch_size, shuffle=True, **options)
    val_loader = DataLoader(SongDataset(val_entries, CONFIG.training.level_idx), batch_size=CONFIG.training.batch_size, shuffle=False, **options)
    dims = ModelDimensions(MERT_HIDDEN_DIM, CONFIG.model.hidden_dim, CONFIG.model.layers, CONFIG.model.kernel_size, CONFIG.model.dropout)
    model = ChartCNN(dims).to(DEVICE)
    print(f"模型参数量={sum(parameter.numel() for parameter in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.training.learning_rate, weight_decay=CONFIG.training.weight_decay, fused=DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.training.lr_t_max, eta_min=CONFIG.training.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss, bad_epochs = float("inf"), 0
    for epoch in range(1, CONFIG.training.num_epochs + 1):
        train_metrics, val_metrics = run_epoch(model, train_loader, optimizer, scaler), run_epoch(model, val_loader, None, scaler)
        scheduler.step()
        metrics = {"train_loss": train_metrics[0], "val_loss": val_metrics[0], "count_loss": val_metrics[1], "duration_loss": val_metrics[2], "exact_frame_accuracy": val_metrics[3] * 100}
        print(f"Epoch {epoch}: train={train_metrics[0]:.4f} val={val_metrics[0]:.4f} count={val_metrics[1]:.4f} duration={val_metrics[2]:.4f} 完整帧准确率={metrics['exact_frame_accuracy']:.2f}% lr={scheduler.get_last_lr()[0]:.2e}")
        state = checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics)
        torch.save(state, CONFIG.paths.checkpoint_dir / "newest.pt")
        if val_metrics[0] < best_loss:
            best_loss, bad_epochs = val_metrics[0], 0
            torch.save(state, CONFIG.paths.checkpoint_dir / "best.pt")
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % CONFIG.training.generation_interval == 0:
            print(f"正式推理: {json.dumps(generation_eval(model, val_entries[:CONFIG.training.val_gen_charts]), ensure_ascii=False)}")
        if bad_epochs >= CONFIG.training.early_stop_patience:
            print("验证损失长期没有改善，提前停止")
            break
    best = torch.load(CONFIG.paths.checkpoint_dir / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    output_entries = train_entries if CONFIG.training.overfit_charts else test_entries
    output_dir = CONFIG.paths.overfit_output_dir if CONFIG.training.overfit_charts else CONFIG.paths.train_output_dir
    print(f"最终正式推理完成: {json.dumps(generation_eval(model, output_entries, output_dir), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
