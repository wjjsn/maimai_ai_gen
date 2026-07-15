"""四头事件 CNN 训练入口；过拟合模式只改变歌曲集合。"""

from pathlib import Path
import shutil

import numpy as np
import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import MERT_HIDDEN_DIM, extract_audio_features
from chart import Chart, Level
from config import CONFIG, checkpoint_config
from dataset import SongDataset, SongEntry, _target_scale, chart_to_targets, collate_songs, discover_songs, parse_level
from generation_metrics import EvaluationItem, build_level_reference, evaluate_generation
from infer import MODEL_KIND, events_to_frames, frames_to_maidata, predict_events, save_inference_files
from model import ChartCNN, ModelDimensions
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, chart_to_tracks


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONSOLE = Console()
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
    evaluation_items = []
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
        if not training:
            predictions = torch.cat((
                output[0].argmax(dim=-1, keepdim=True),
                output[1].argmax(dim=-1, keepdim=True),
                _duration_frames(output[2]),
            ), dim=-1).cpu().numpy().astype(np.int32)
            targets = (batch["events"] * torch.tensor(_target_scale(), device=DEVICE)).round().cpu().numpy().astype(np.int32)
            masks = batch["mask"].cpu().numpy()
            for prediction, target, valid, entry in zip(predictions, targets, masks, batch["entries"]):
                prediction, target = prediction[valid], target[valid]
                frames, song_stats = events_to_frames(prediction)
                generated = Chart(all_levels=[None] * 7)
                generated.all_levels[entry.level_idx] = Level("generated", entry.level_query, frames)
                postprocessed = chart_to_tracks(generated, entry.level_idx)
                postprocessed = np.pad(postprocessed, ((0, max(0, len(prediction) - len(postprocessed))), (0, 0)))[:len(prediction)]
                evaluation_items.append(EvaluationItem(
                    f"{entry.chart_path.parent.name} level={entry.level_idx} ds={entry.level_query:g}",
                    entry.level_query, target, postprocessed, song_stats["dropped"],
                ))
        totals += (loss.item(), count_loss.item(), duration_loss.item())
    return tuple((totals / max(len(loader), 1)).tolist()), evaluation_items


@torch.no_grad()
def generation_eval(model, entries, reference=None, output_dir: Path | None = None):
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
            stats["comparison"].append(EvaluationItem(label, entry.level_query, target, predicted, song_stats["dropped"]))
            if output_dir is not None:
                difficulty_dir = output_dir / relative / f"level_{entry.level_idx}_{entry.level_query:g}"
                save_inference_files(entry.audio_path, frames_to_maidata(frames, entry.level_idx, entry.level_query, relative.name), difficulty_dir)
    stats["metrics"] = evaluate_generation(stats["comparison"], reference)
    stats["score"] = stats["metrics"]["score"]
    return stats


def build_reference(entries) -> dict[float, dict]:
    items = []
    texts = {}
    for entry in tqdm(entries, desc="统计精确定数基准", leave=False):
        text = texts.setdefault(entry.chart_path, entry.chart_path.read_text(encoding="utf-8"))
        items.append((entry.level_query, chart_to_tracks(parse_level(text, entry.level_idx), entry.level_idx)))
    return build_level_reference(items)


def checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics, training_state, data_info):
    return {
        "checkpoint_version": 7,
        "evaluation_version": 2,
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
        or state.get("evaluation_version") != 2
        or state.get("model_kind") != MODEL_KIND
        or state.get("config") != checkpoint_config()
        or state.get("dims") != dims
        or state.get("data_info") != data_info
    ):
        raise ValueError("检查点的架构、配置或难度数据快照与当前训练不一致")


def format_epoch_log(epoch: int, train_metrics, val_metrics, learning_rate: float, saved_best: bool, bad_epochs: int):
    table = Table(title=f"第 {epoch} 轮训练", show_header=True, header_style="bold cyan")
    for column in ("训练损失", "验证损失", "数量损失", "时长损失", "学习率", "生成未改善"):
        table.add_column(column, justify="right")
    style = "bold green" if bad_epochs == 0 else ("red" if bad_epochs >= CONFIG.training.early_stop_patience - 2 else "yellow")
    table.add_row(
        f"{train_metrics[0]:.5f}", f"{val_metrics[0]:.5f}", f"{val_metrics[1]:.5f}",
        f"{val_metrics[2]:.5f}", f"{learning_rate:.2e}", Text(str(bad_epochs), style=style),
    )
    status = Text("验证损失新低，已保存 best.pt", style="bold green") if saved_best else Text("验证损失未刷新", style="dim")
    return Group(table, status)


def _deviation_style(value: float, reference: float, samples: int) -> str:
    if samples < 3:
        return "yellow"
    difference = abs(value / reference - 1) if reference else float("inf")
    return "green" if difference <= 0.10 else "yellow" if difference <= 0.25 else "red"


def _deviation(values, references, samples: int) -> Text:
    average, _middle = values
    reference_average, _reference_middle = references
    difference = (average / reference_average - 1) * 100 if reference_average else 0.0
    return Text(f"{difference:+.1f}%", style=_deviation_style(average, reference_average, samples))


def _accuracy(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "无样本"


def format_generation_log(label: str, stats: dict[str, int], saved_best: bool = False, show_levels: bool = True):
    metrics = stats["metrics"]
    summary = Table(show_header=True, header_style="bold cyan")
    for column in ("综合 F1", "Tap F1", "持续起点 F1", "严格 F1", "时长 MAE", "数量偏差", "过滤率", "空谱率"):
        summary.add_column(column, justify="right")
    score_style = "bold green" if saved_best else "white"
    duration = (
        f"{metrics['duration_mae_frames'] / CONFIG.audio.frames_per_sec * 1000:.0f} ms"
        if metrics["matched_durations"] else "无匹配"
    )
    summary.add_row(
        Text(f"{metrics['score']:.1%}", style=score_style), f"{metrics['tap']['f1']:.1%}",
        f"{metrics['long']['f1']:.1%}", f"{metrics['strict_score']:.1%}",
        duration,
        f"{metrics['count_error']:+.1%}", f"{metrics['dropped_rate']:.1%}", f"{metrics['empty_rate']:.1%}",
    )

    levels = Table(title="精确定数生成分布（生成/完整数据集基准）", header_style="bold cyan")
    for column in ("DS", "验证", "基准", "Tap 偏差", "持续偏差", "总数偏差", "间隔偏差", "持续类型准确率", "持续时长准确率", "F1"):
        levels.add_column(column, justify="right")
    for row in metrics["levels"]:
        samples = row["charts"]
        levels.add_row(
            f"{row['level_query']:.1f}", str(samples), str(row["reference_charts"]),
            _deviation(row["generated_tap"], row["target_tap"], samples),
            _deviation(row["generated_long"], row["target_long"], samples),
            _deviation(row["generated_total"], row["target_total"], samples),
            _deviation(row["generated_interval"], row["target_interval"], samples),
            _accuracy(row["long_type_accuracy"]),
            _accuracy(row["long_duration_accuracy"]),
            f"{row['score']:.1%}",
        )
    subtitle = Text(
        f"{stats['songs']} 首 / {stats['charts']} 张谱面 / 10 ms 容差",
        style="dim",
    )
    if saved_best:
        subtitle.append(" / 新最佳生成模型", style="bold green")
    content = Group(subtitle, summary, levels) if show_levels else Group(subtitle, summary)
    return Panel(content, title=label, border_style="green" if saved_best else "cyan")


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
    state = {"checkpoint_version": 7, "evaluation_version": 2, "model_kind": MODEL_KIND, "config": checkpoint_config(), "dims": dims, "data_info": {"schema_version": 1, "digest": "test"}}
    validate_checkpoint(state, dims, state["data_info"])
    try:
        validate_checkpoint(state, dims, {"schema_version": 1, "digest": "changed"})
    except ValueError:
        pass
    else:
        raise AssertionError("难度数据快照变化时必须拒绝检查点")
    console = Console(record=True, width=140, color_system=None)
    console.print(format_epoch_log(1, (0.1, 0.1, 0.0), (0.1, 0.1, 0.0), 0.001, True, 0))
    assert "验证损失新低" in console.export_text() and "完整时间帧命中率" not in console.export_text()
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
    CONSOLE.print(f"设备={DEVICE}，训练/验证/测试歌曲={song_count(train_entries)}/{song_count(val_entries)}/{song_count(test_entries)} 首，谱面={len(train_entries)}/{len(val_entries)}/{len(test_entries)} 张")
    if CONFIG.training.overfit_charts:
        CONSOLE.print(Panel(
            "训练集与验证集是同一批歌曲，只能衡量记忆能力，不能代表未见歌曲上的泛化效果。",
            title="OVERFIT 模式", border_style="bold red",
        ))
    reference = build_reference(entries)
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
        train_metrics, _ = run_epoch(model, train_loader, optimizer, scaler)
        val_metrics, validation_items = run_epoch(model, val_loader, None, scaler)
        generation_metrics = evaluate_generation(validation_items, reference)
        scheduler.step()
        metrics = {
            "train_loss": train_metrics[0], "val_loss": val_metrics[0],
            "count_loss": val_metrics[1], "duration_loss": val_metrics[2],
            "generation_score": generation_metrics["score"],
        }
        if val_metrics[0] < best_loss:
            best_loss = val_metrics[0]
            saved_best = True
        else:
            saved_best = False
        if generation_metrics["score"] > best_gen_score:
            best_gen_score, bad_epochs = generation_metrics["score"], 0
            saved_best_gen = True
        else:
            bad_epochs += 1
            saved_best_gen = False
        CONSOLE.print(format_epoch_log(epoch, train_metrics, val_metrics, scheduler.get_last_lr()[0], saved_best, bad_epochs))
        validation_stats = {
            "songs": song_count(val_entries), "charts": len(validation_items),
            "metrics": generation_metrics,
        }
        CONSOLE.print(format_generation_log(
            "完整验证集实际生成效果", validation_stats, saved_best_gen,
            show_levels=epoch == 1 or epoch % CONFIG.training.generation_interval == 0,
        ))
        training_state = {"best_loss": best_loss, "best_gen_score": best_gen_score, "bad_epochs": bad_epochs}
        state = checkpoint(model, optimizer, scheduler, scaler, dims, epoch, metrics, training_state, data_info)
        torch.save(state, CONFIG.paths.checkpoint_dir / "newest.pt")
        if saved_best:
            torch.save(state, CONFIG.paths.checkpoint_dir / "best.pt")
        if saved_best_gen:
            torch.save(state, best_gen_path)
        if bad_epochs >= CONFIG.training.early_stop_patience:
            CONSOLE.print("[bold red]完整验证集综合 F1 长期没有改善，提前停止[/bold red]")
            break
    if not best_gen_path.is_file():
        raise ValueError(f"没有可用的最佳生成检查点：{best_gen_path}")
    best = torch.load(best_gen_path, map_location=DEVICE, weights_only=False)
    validate_checkpoint(best, dims, data_info)
    model.load_state_dict(best["model_state_dict"])
    output_entries = train_entries if CONFIG.training.overfit_charts else test_entries
    output_dir = CONFIG.paths.overfit_output_dir if CONFIG.training.overfit_charts else CONFIG.paths.train_output_dir
    CONSOLE.print(format_generation_log("最终整曲推理完成", generation_eval(model, output_entries, reference, output_dir)))


if __name__ == "__main__":
    main()
