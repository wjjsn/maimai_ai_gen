"""四头事件 CNN 训练入口；过拟合模式只改变歌曲集合。"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import MERT_HIDDEN_DIM, extract_audio_features
from chart import Chart, Level
from config import CONFIG, checkpoint_config
from dataset import SongDataset, _target_scale, chart_to_targets, collate_songs, discover_songs
from generation_metrics import format_generation_comparison, generation_score
from infer import MODEL_KIND, events_to_frames, frames_to_maidata, predict_events, save_inference_files
from maidata_parser import parse_maidata
from model import ChartCNN, ModelDimensions
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, chart_to_tracks


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


def _duration_target(events: torch.Tensor) -> torch.Tensor:
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    return torch.log1p(events[..., HOLD_DURATION_1:] * maximum) / np.log1p(maximum)


def _duration_frames(output: torch.Tensor) -> torch.Tensor:
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    return torch.expm1(output * np.log1p(maximum)).round()


def compute_loss(output, batch):
    target, mask = batch["events"], batch["mask"]
    tap_logits, hold_logits, durations = output
    tap_target = (target[..., TAP_COUNT] * 8).round().long()
    hold_target = (target[..., HOLD_START_COUNT] * 2).round().long()
    count_wrong = ((tap_logits.argmax(dim=-1) != tap_target) | (hold_logits.argmax(dim=-1) != hold_target)).detach()
    wrong_weight = 1 + count_wrong * (CONFIG.training.wrong_loss_weight - 1)
    count_active = target[..., :2].amax(dim=-1) > 0
    count_weight = 1 + count_active * (CONFIG.training.short_loss_weight - 1)
    weights = count_weight * wrong_weight * mask
    tap_loss = nn.functional.cross_entropy(tap_logits.transpose(1, 2), tap_target, reduction="none")
    hold_loss = nn.functional.cross_entropy(hold_logits.transpose(1, 2), hold_target, reduction="none")
    count_loss = ((tap_loss + hold_loss) * weights).sum() / weights.sum().clamp_min(1)
    hold_count = hold_target
    duration_mask = torch.stack((hold_count >= 1, hold_count >= 2), dim=-1) * mask.unsqueeze(-1)
    duration_error = nn.functional.smooth_l1_loss(durations, _duration_target(target), reduction="none")
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
        prediction = torch.cat((output[0].argmax(dim=-1, keepdim=True), output[1].argmax(dim=-1, keepdim=True), _duration_frames(output[2])), dim=-1)
        target = torch.cat(((batch["events"][..., :2] * torch.tensor((8, 2), device=DEVICE)).round(), batch["events"][..., HOLD_DURATION_1:] * round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)), dim=-1)
        duration_match = ((prediction[..., HOLD_DURATION_1:] == target[..., HOLD_DURATION_1:]) | (target[..., HOLD_DURATION_1:] == 0)).all(dim=-1)
        correct += ((prediction[..., :HOLD_DURATION_1] == target[..., :HOLD_DURATION_1]).all(dim=-1) & duration_match & batch["mask"]).sum().item()
        frames += batch["mask"].sum().item()
        totals += (loss.item(), count_loss.item(), duration_loss.item())
    return (*((totals / max(len(loader), 1)).tolist()), correct / max(frames, 1))


@torch.no_grad()
def generation_eval(model, entries, output_dir: Path | None = None):
    stats = {"songs": 0, "events": 0, "dropped": 0, "comparison": []}
    for entry in tqdm(entries, desc="正式整曲推理", leave=False):
        events = predict_events(model, extract_audio_features(entry.audio_path, DEVICE), DEVICE)
        frames, song_stats = events_to_frames(events)
        generated = Chart(all_levels=[None] * 7)
        generated.all_levels[CONFIG.training.level_idx] = Level("generated", entry.level_query, frames)
        # 正式解码允许末尾持续音超过音频；比较时间轴只覆盖当前音频。
        predicted = chart_to_tracks(generated, CONFIG.training.level_idx)
        predicted = np.pad(predicted, ((0, max(0, len(events) - len(predicted))), (0, 0)))[:len(events)]
        source = parse_maidata(entry.chart_path.read_text(encoding="utf-8"))
        target = (chart_to_targets(source, len(events), CONFIG.training.level_idx) * _target_scale()).round().astype(np.int32)
        stats["songs"] += 1
        stats["events"] += song_stats["events"]
        stats["dropped"] += song_stats["dropped"]
        stats["comparison"].append((str(entry.chart_path.parent.relative_to(CONFIG.paths.charts_dir)), target, predicted, song_stats["dropped"]))
        if output_dir is not None:
            relative = entry.chart_path.parent.relative_to(CONFIG.paths.charts_dir)
            save_inference_files(entry.audio_path, frames_to_maidata(frames, CONFIG.training.level_idx, relative.name), output_dir / relative)
    stats["score"] = generation_score(stats["comparison"])
    return stats


def checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics):
    return {"checkpoint_version": 6, "model_kind": MODEL_KIND, "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(), "dims": dims, "config": checkpoint_config(), "metrics": metrics}


def format_epoch_log(epoch: int, train_metrics, val_metrics, learning_rate: float, saved_best: bool, bad_epochs: int) -> str:
    """将一轮训练的数字拆成可直接阅读的说明。"""
    best_text = "这轮验证损失刷新最低记录，已保存为 checkpoints/best.pt。" if saved_best else f"这轮没有刷新最低验证损失；已经连续 {bad_epochs} 轮未改善。"
    return "\n".join([
        "",
        f"========== 第 {epoch} 轮训练完成 ==========" ,
        "训练集结果（训练集会使用随机数据增强，因此不应直接和验证集逐项比较）：",
        f"  训练总损失：{train_metrics[0]:.6f}。这是数量预测损失和长按时长预测损失之和，越低表示训练数据上的回归误差越小。",
        "验证集结果（不使用数据增强，也不会更新模型参数，用它决定最佳模型）：",
        f"  验证总损失：{val_metrics[0]:.6f}。这是选择最佳 checkpoint 和提前停止时使用的主要指标，越低越好。",
        f"  音符数量预测损失：{val_metrics[1]:.6f}。它衡量每个 5 毫秒时间帧中 Tap/Touch 数量和长按起点数量是否接近真实谱面；预测成错误整数的帧会按配置加重。",
        f"  长按时长预测损失：{val_metrics[2]:.6f}。它只在真实存在长按或 Slide 的起点计算，衡量预测持续时间与真实持续时间的差距。",
        f"  完整时间帧命中率：{val_metrics[3] * 100:.2f}%。只有 Tap/Touch 数量、长按起点数量，以及真实存在的长按时长都完全正确时，该 5 毫秒帧才算命中。",
        f"  当前学习率：{learning_rate:.6e}。这是下一轮参数更新使用的步长，余弦退火会让它逐步降低。",
        f"检查点状态：{best_text}",
        "================================",
    ])


def format_generation_log(label: str, stats: dict[str, int]) -> str:
    return "\n".join([
        "",
        f"---------- {label} ----------",
        f"参与整曲推理的歌曲数：{stats['songs']} 首。",
        f"成功写入生成谱面的音符事件总数：{stats['events']} 个。这个数字只表示模型生成量，不代表与原谱面的匹配程度。",
        f"因为短音之间少于最小间隔而被丢弃的 Tap/Touch 数量：{stats['dropped']} 个。该数值很高通常表示模型把短音生成得过密。",
        f"正式生成选模 F1：{stats['score'] * 100:.2f}%。这是 Tap 与持续音起点微平均 F1 的均值，越高越好。",
        *format_generation_comparison(stats["comparison"]),
        "--------------------------------",
    ])


def _self_check() -> None:
    batch = {
        "events": torch.zeros(1, 2, 4, device=DEVICE),
        "mask": torch.ones(1, 2, dtype=torch.bool, device=DEVICE),
    }
    model = ChartCNN(ModelDimensions(4, 8, 1, 3, 0)).to(DEVICE)
    output = model(torch.zeros(1, 2, 4, device=DEVICE), batch["mask"])
    loss, count_loss, duration_loss = compute_loss(output, batch)
    assert loss.isfinite() and count_loss.isfinite() and duration_loss.isfinite()
    batch["events"][0, 0] = torch.tensor((0, 0.5, 0.1, 0))
    assert compute_loss(output, batch)[2] > 0
    short = np.zeros((1, 4), dtype=np.int32)
    aligned = np.pad(short, ((0, 2), (0, 0)))[:3]
    assert aligned.shape == (3, 4) and not aligned.any()
    assert "完整时间帧命中率" in format_epoch_log(1, (0.1, 0.1, 0.0, 0.0), (0.1, 0.1, 0.0, 0.0), 0.001, True, 0)
    print("[train] 自检通过")


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    print(f"开始扫描训练歌曲：{CONFIG.paths.charts_dir}", flush=True)
    entries, skipped = discover_songs(CONFIG.paths.charts_dir, CONFIG.training.level_idx)
    if skipped:
        print(f"跳过 {len(skipped)} 首含非默认等待 Slide、长按溢出或无效数据的歌曲", flush=True)
    minimum = CONFIG.training.overfit_charts or 3
    if len(entries) < minimum:
        raise ValueError(f"有效歌曲只有 {len(entries)} 首，需要至少 {minimum} 首")
    train_entries, val_entries, test_entries = split_entries(entries, CONFIG.training.seed, CONFIG.training.overfit_charts)
    print(f"设备={DEVICE}，训练/验证/测试={len(train_entries)}/{len(val_entries)}/{len(test_entries)} 首", flush=True)
    options = {"num_workers": CONFIG.training.num_workers, "pin_memory": CONFIG.training.pin_memory, "collate_fn": collate_songs}
    if CONFIG.training.num_workers:
        options["prefetch_factor"] = CONFIG.training.prefetch_factor
    train_loader = DataLoader(SongDataset(train_entries, CONFIG.training.level_idx, augment=True), batch_size=CONFIG.training.batch_size, shuffle=True, **options)
    val_loader = DataLoader(SongDataset(val_entries, CONFIG.training.level_idx), batch_size=CONFIG.training.batch_size, shuffle=False, **options)
    dims = ModelDimensions(MERT_HIDDEN_DIM, CONFIG.model.hidden_dim, CONFIG.model.layers, CONFIG.model.kernel_size, CONFIG.model.dropout)
    model = ChartCNN(dims).to(DEVICE)
    print(f"模型参数量={sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.training.learning_rate, weight_decay=CONFIG.training.weight_decay, fused=DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.training.lr_t_max, eta_min=CONFIG.training.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss, best_gen_score, bad_epochs = float("inf"), -1.0, 0
    for epoch in range(1, CONFIG.training.num_epochs + 1):
        if epoch == 1:
            print("开始现场解码音频并提取首批 MERT 特征", flush=True)
        train_metrics, val_metrics = run_epoch(model, train_loader, optimizer, scaler), run_epoch(model, val_loader, None, scaler)
        scheduler.step()
        metrics = {"train_loss": train_metrics[0], "val_loss": val_metrics[0], "count_loss": val_metrics[1], "duration_loss": val_metrics[2], "exact_frame_accuracy": val_metrics[3] * 100}
        state = checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics)
        torch.save(state, CONFIG.paths.checkpoint_dir / "newest.pt")
        if val_metrics[0] < best_loss:
            best_loss, bad_epochs = val_metrics[0], 0
            torch.save(state, CONFIG.paths.checkpoint_dir / "best.pt")
            saved_best = True
        else:
            bad_epochs += 1
            saved_best = False
        print(format_epoch_log(epoch, train_metrics, val_metrics, scheduler.get_last_lr()[0], saved_best, bad_epochs))
        if epoch == 1 or epoch % CONFIG.training.generation_interval == 0:
            generated = generation_eval(model, val_entries[:CONFIG.training.val_gen_charts])
            if generated["score"] > best_gen_score:
                best_gen_score = generated["score"]
                torch.save(state, CONFIG.paths.checkpoint_dir / "best_gen.pt")
            print(format_generation_log("定期整曲推理检查", generated))
        if bad_epochs >= CONFIG.training.early_stop_patience:
            print("验证损失长期没有改善，提前停止")
            break
    best = torch.load(CONFIG.paths.checkpoint_dir / "best_gen.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    output_entries = train_entries if CONFIG.training.overfit_charts else test_entries
    output_dir = CONFIG.paths.overfit_output_dir if CONFIG.training.overfit_charts else CONFIG.paths.train_output_dir
    print(format_generation_log("最终整曲推理完成", generation_eval(model, output_entries, output_dir)))


if __name__ == "__main__":
    main()
