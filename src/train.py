"""四头事件 CNN 训练入口；过拟合模式只改变歌曲集合。"""

from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import MERT_HIDDEN_DIM, extract_audio_features
from chart import Chart, Level
from config import CONFIG, checkpoint_config
from dataset import SongDataset, SongEntry, _target_scale, chart_to_targets, collate_songs, discover_songs, parse_level
from generation_metrics import format_generation_comparison, generation_score
from infer import MODEL_KIND, events_to_frames, frames_to_maidata, predict_events, save_inference_files
from model import ChartCNN, ModelDimensions
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, chart_to_tracks


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def split_entries(entries, seed: int, overfit_charts: int):
    songs = {}
    for entry in entries:
        songs.setdefault(entry.song_key, []).append(entry)
    keys = list(songs)
    order = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [keys[i] for i in order]
    if overfit_charts:
        selected = [entry for path in shuffled[:overfit_charts] for entry in songs[path]]
        return selected, selected, []
    val_count, test_count = max(1, round(len(keys) * CONFIG.training.val_ratio)), max(1, round(len(keys) * CONFIG.training.test_ratio))
    groups = shuffled[val_count + test_count:], shuffled[:val_count], shuffled[val_count:val_count + test_count]
    return tuple([entry for path in group for entry in songs[path]] for group in groups)


def first_songs(entries, count: int):
    selected, keys = [], set()
    for entry in entries:
        if entry.song_key not in keys:
            if len(keys) == count:
                break
            keys.add(entry.song_key)
        selected.append(entry)
    return selected


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
        for key in ("features", "events", "mask", "level_queries", "level_indices"):
            batch[key] = batch[key].to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            output = model(batch["features"], batch["level_queries"], batch["level_indices"], batch["mask"])
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
    stats = {"songs": 0, "charts": 0, "events": 0, "dropped": 0, "comparison": []}
    groups = {}
    for entry in entries:
        groups.setdefault(entry.chart_path, []).append(entry)
    for group in tqdm(groups.values(), desc="正式整曲推理", leave=False):
        features = extract_audio_features(group[0].audio_path, DEVICE)
        text = group[0].chart_path.read_text(encoding="utf-8")
        stats["songs"] += 1
        for entry in group:
            events = predict_events(model, features, entry.level_query, entry.level_idx, DEVICE)
            frames, song_stats = events_to_frames(events)
            generated = Chart(all_levels=[None] * 7)
            generated.all_levels[entry.level_idx] = Level("generated", entry.level_query, frames)
            # 正式解码允许末尾持续音超过音频；比较时间轴只覆盖当前音频。
            predicted = chart_to_tracks(generated, entry.level_idx)
            predicted = np.pad(predicted, ((0, max(0, len(events) - len(predicted))), (0, 0)))[:len(events)]
            source = parse_level(text, entry.level_idx)
            target = (chart_to_targets(source, len(events), entry.level_idx) * _target_scale()).round().astype(np.int32)
            stats["charts"] += 1
            stats["events"] += song_stats["events"]
            stats["dropped"] += song_stats["dropped"]
            relative = entry.chart_path.parent.relative_to(CONFIG.paths.charts_dir)
            label = f"{relative} level={entry.level_idx} ds={entry.level_query:g}"
            stats["comparison"].append((label, target, predicted, song_stats["dropped"]))
            if output_dir is not None:
                difficulty_dir = output_dir / relative / f"level_{entry.level_idx}_{entry.level_query:g}"
                save_inference_files(entry.audio_path, frames_to_maidata(frames, entry.level_idx, entry.level_query, relative.name), difficulty_dir)
    stats["score"] = generation_score(stats["comparison"])
    return stats


def checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics, training_state, data_info):
    return {
        "checkpoint_version": 7,
        "model_kind": MODEL_KIND,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "dims": dims,
        "config": checkpoint_config(),
        "data_info": data_info,
        "training_state": training_state,
        "metrics": metrics,
    }


def validate_checkpoint(state, dims, data_info) -> None:
    if (
        state.get("checkpoint_version") != 7
        or state.get("model_kind") != MODEL_KIND
        or state.get("config") != checkpoint_config()
        or state.get("dims") != dims
        or state.get("data_info") != data_info
    ):
        raise ValueError("检查点的架构、配置或难度数据快照与当前训练不一致")


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
        f"参与整曲推理的谱面数：{stats['charts']} 张。",
        f"成功写入生成谱面的音符事件总数：{stats['events']} 个。这个数字只表示模型生成量，不代表与原谱面的匹配程度。",
        f"因为短音之间少于最小间隔而被丢弃的 Tap/Touch 数量：{stats['dropped']} 个。该数值很高通常表示模型把短音生成得过密。",
        f"正式生成选模 F1：{stats['score'] * 100:.2f}%。这是 Tap 与持续音起点微平均 F1 的均值，越高越好。",
        *format_generation_comparison(stats["comparison"]),
        "--------------------------------",
    ])


def _self_check() -> None:
    from unittest.mock import patch

    batch = {
        "events": torch.zeros(1, 2, 4, device=DEVICE),
        "mask": torch.ones(1, 2, dtype=torch.bool, device=DEVICE),
    }
    model = ChartCNN(ModelDimensions(4, 8, 1, 3, 0)).to(DEVICE)
    output = model(torch.zeros(1, 2, 4, device=DEVICE), torch.tensor([13.7], device=DEVICE), torch.tensor([5], device=DEVICE), batch["mask"])
    loss, count_loss, duration_loss = compute_loss(output, batch)
    assert loss.isfinite() and count_loss.isfinite() and duration_loss.isfinite()
    batch["events"][0, 0] = torch.tensor((0, 0.5, 0.1, 0))
    assert compute_loss(output, batch)[2] > 0
    short = np.zeros((1, 4), dtype=np.int32)
    aligned = np.pad(short, ((0, 2), (0, 0)))[:3]
    assert aligned.shape == (3, 4) and not aligned.any()
    entries = [
        SongEntry(Path("a/maidata.txt"), Path("a/track.mp3"), "same", "1", 2, 4.0),
        SongEntry(Path("a/maidata.txt"), Path("a/track.mp3"), "same", "1", 5, 13.0),
        SongEntry(Path("b/maidata.txt"), Path("b/track.mp3"), "same", "10001", 5, 14.0),
        SongEntry(Path("c/maidata.txt"), Path("c/track.mp3"), "other", "2", 5, 14.5),
        SongEntry(Path("d/maidata.txt"), Path("d/track.mp3"), "third", "3", 5, 14.7),
    ]
    train, val, test = split_entries(entries, 42, 0)
    assert not ({entry.song_key for entry in train} & {entry.song_key for entry in val + test})
    assert len(first_songs(entries, 1)) == 3
    source = Chart(all_levels=[None] * 7)
    source.all_levels[2] = Level("basic", 4.0, [])
    source.all_levels[5] = Level("master", 13.0, [])
    eval_path = CONFIG.paths.charts_dir / "self-check" / "maidata.txt"
    eval_audio = eval_path.parent / "track.mp3"
    eval_entries = [
        SongEntry(eval_path, eval_audio, "same", "1", 2, 4.0),
        SongEntry(eval_path, eval_audio, "same", "1", 5, 13.0),
    ]
    with (
        patch("train.extract_audio_features", return_value=np.zeros((3, 4), dtype=np.float32)) as extract,
        patch("train.predict_events", return_value=np.zeros((3, 4), dtype=np.int32)),
        patch("train.parse_level", side_effect=lambda _text, level_idx: source),
        patch.object(Path, "read_text", return_value=""),
    ):
        stats = generation_eval(model, eval_entries)
    assert extract.call_count == 1 and stats["songs"] == 1 and stats["charts"] == 2

    dims = ModelDimensions(4, 8, 1, 3, 0)
    state = {"checkpoint_version": 7, "model_kind": MODEL_KIND, "config": checkpoint_config(), "dims": dims, "data_info": {"schema_version": 1, "digest": "test"}}
    validate_checkpoint(state, dims, state["data_info"])
    try:
        validate_checkpoint(state, dims, {"schema_version": 1, "digest": "changed"})
    except ValueError:
        pass
    else:
        raise AssertionError("难度数据快照变化时必须拒绝检查点")
    assert "完整时间帧命中率" in format_epoch_log(1, (0.1, 0.1, 0.0, 0.0), (0.1, 0.1, 0.0, 0.0), 0.001, True, 0)
    print("[train] 自检通过")


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    print(f"开始扫描训练歌曲：{CONFIG.paths.charts_dir}", flush=True)
    entries, skipped, data_info = discover_songs(CONFIG.paths.charts_dir)
    if skipped:
        print(f"忽略 {len(skipped)} 个无法匹配或无效的歌曲/难度项", flush=True)
    minimum = CONFIG.training.overfit_charts or 3
    songs = {entry.song_key for entry in entries}
    if len(songs) < minimum:
        raise ValueError(f"有效歌曲只有 {len(songs)} 首，需要至少 {minimum} 首")
    train_entries, val_entries, test_entries = split_entries(entries, CONFIG.training.seed, CONFIG.training.overfit_charts)
    song_count = lambda group: len({entry.song_key for entry in group})
    print(f"设备={DEVICE}，训练/验证/测试歌曲={song_count(train_entries)}/{song_count(val_entries)}/{song_count(test_entries)} 首，谱面={len(train_entries)}/{len(val_entries)}/{len(test_entries)} 张", flush=True)
    options = {"num_workers": CONFIG.training.num_workers, "pin_memory": CONFIG.training.pin_memory, "collate_fn": collate_songs}
    if CONFIG.training.num_workers:
        options["prefetch_factor"] = CONFIG.training.prefetch_factor
    train_loader = DataLoader(SongDataset(train_entries, augment=True), batch_size=CONFIG.training.batch_size, shuffle=True, **options)
    val_loader = DataLoader(SongDataset(val_entries), batch_size=CONFIG.training.batch_size, shuffle=False, **options)
    dims = ModelDimensions(MERT_HIDDEN_DIM, CONFIG.model.hidden_dim, CONFIG.model.layers, CONFIG.model.kernel_size, CONFIG.model.dropout)
    model = ChartCNN(dims).to(DEVICE)
    print(f"模型参数量={sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.training.learning_rate, weight_decay=CONFIG.training.weight_decay, fused=DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.training.lr_t_max, eta_min=CONFIG.training.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss, best_gen_score, bad_epochs = float("inf"), -1.0, 0
    start_epoch = 1
    best_gen_path = CONFIG.paths.checkpoint_dir / "best_gen.pt"
    if CONFIG.training.resume_checkpoint is not None:
        print(f"从检查点恢复训练：{CONFIG.training.resume_checkpoint}", flush=True)
        state = torch.load(CONFIG.training.resume_checkpoint, map_location=DEVICE, weights_only=False)
        validate_checkpoint(state, dims, data_info)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        training_state = state.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError("恢复检查点缺少训练状态")
        best_loss = training_state["best_loss"]
        best_gen_score = training_state["best_gen_score"]
        bad_epochs = training_state["bad_epochs"]
        source_best_gen = CONFIG.training.resume_checkpoint.parent / "best_gen.pt"
        if not source_best_gen.is_file():
            raise ValueError(f"恢复训练缺少配套最佳生成检查点：{source_best_gen}")
        best_gen = torch.load(source_best_gen, map_location="cpu", weights_only=False)
        validate_checkpoint(best_gen, dims, data_info)
        best_gen_state = best_gen.get("training_state")
        if not isinstance(best_gen_state, dict) or best_gen_state.get("best_gen_score") != best_gen_score:
            raise ValueError("配套最佳生成检查点与恢复检查点的最佳分数不一致")
        if source_best_gen.resolve() != best_gen_path.resolve():
            shutil.copy2(source_best_gen, best_gen_path)
        print(f"已恢复到第 {state['epoch']} 轮，从第 {start_epoch} 轮继续训练", flush=True)
    for epoch in range(start_epoch, CONFIG.training.num_epochs + 1):
        if epoch == 1:
            print("开始现场解码音频并提取首批 MERT 特征", flush=True)
        train_metrics, val_metrics = run_epoch(model, train_loader, optimizer, scaler), run_epoch(model, val_loader, None, scaler)
        scheduler.step()
        metrics = {"train_loss": train_metrics[0], "val_loss": val_metrics[0], "count_loss": val_metrics[1], "duration_loss": val_metrics[2], "exact_frame_accuracy": val_metrics[3] * 100}
        if val_metrics[0] < best_loss:
            best_loss, bad_epochs = val_metrics[0], 0
            saved_best = True
        else:
            bad_epochs += 1
            saved_best = False
        print(format_epoch_log(epoch, train_metrics, val_metrics, scheduler.get_last_lr()[0], saved_best, bad_epochs))
        if epoch == 1 or epoch % CONFIG.training.generation_interval == 0:
            generated = generation_eval(model, first_songs(val_entries, CONFIG.training.val_gen_charts))
            if generated["score"] > best_gen_score:
                best_gen_score = generated["score"]
                saved_best_gen = True
            else:
                saved_best_gen = False
            print(format_generation_log("定期整曲推理检查", generated))
        else:
            saved_best_gen = False
        training_state = {"best_loss": best_loss, "best_gen_score": best_gen_score, "bad_epochs": bad_epochs}
        state = checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics, training_state, data_info)
        torch.save(state, CONFIG.paths.checkpoint_dir / "newest.pt")
        if saved_best:
            torch.save(state, CONFIG.paths.checkpoint_dir / "best.pt")
        if saved_best_gen:
            torch.save(state, best_gen_path)
        if bad_epochs >= CONFIG.training.early_stop_patience:
            print("验证损失长期没有改善，提前停止")
            break
    if not best_gen_path.is_file():
        raise ValueError(f"没有可用的最佳生成检查点：{best_gen_path}")
    best = torch.load(best_gen_path, map_location=DEVICE, weights_only=False)
    validate_checkpoint(best, dims, data_info)
    model.load_state_dict(best["model_state_dict"])
    output_entries = train_entries if CONFIG.training.overfit_charts else test_entries
    output_dir = CONFIG.paths.overfit_output_dir if CONFIG.training.overfit_charts else CONFIG.paths.train_output_dir
    print(format_generation_log("最终整曲推理完成", generation_eval(model, output_entries, output_dir)))


if __name__ == "__main__":
    main()
