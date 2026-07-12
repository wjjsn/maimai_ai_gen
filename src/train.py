from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import Whisper, ModelDimensions
from config import CONFIG, checkpoint_config, validate_checkpoint_config
from dataset import AudioAugmentedDataset, ChartDataset, RotatedDataset, collate_segments
from content_metrics import content_match_frame_counts
from infer import decode_segment, overlap_infer
from maidata_parser import compiler
from mert_cache import feature_signature
from tokenizer import EOS, FRAME_END, FRAME_START, PAD, SOS, VOCAB_SIZE

# ─────────────────────── 0. 配置 ───────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {DEVICE}")

CHARTS_DIR = CONFIG.paths.charts_dir
LEVEL_IDX = CONFIG.training.level_idx
MAX_TOKENS = CONFIG.model.max_tokens
BATCH_SIZE = CONFIG.training.batch_size
NUM_WORKERS = CONFIG.training.num_workers
PREFETCH_FACTOR = CONFIG.training.prefetch_factor
PIN_MEMORY = CONFIG.training.pin_memory
NUM_EPOCHS = CONFIG.training.num_epochs
LR = CONFIG.training.learning_rate
WEIGHT_DECAY = CONFIG.training.weight_decay
VAL_RATIO = CONFIG.training.val_ratio
GRAD_CLIP = CONFIG.training.grad_clip
EARLY_STOP_PATIENCE = CONFIG.training.early_stop_patience
LABEL_SMOOTHING = CONFIG.training.label_smoothing
EOS_LOSS_WEIGHT = CONFIG.training.eos_loss_weight
LR_T_MAX = CONFIG.training.lr_t_max
SEED = CONFIG.training.seed
VAL_GEN_CHARTS = CONFIG.training.val_gen_charts
VAL_ORACLE_WINDOWS = CONFIG.training.val_oracle_windows
OVERFIT_CHARTS = CONFIG.training.overfit_charts
RESUME_PATH = CONFIG.training.resume_path
CHECKPOINT_DIR = CONFIG.paths.checkpoint_dir
if CHECKPOINT_DIR.exists() and not CHECKPOINT_DIR.is_dir():
    raise ValueError(f"检查点路径不是目录: {CHECKPOINT_DIR}")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────── 1. 模型 ───────────────────────

dims = ModelDimensions(
    n_audio_ctx=CONFIG.window.mert_frames,
    n_vocab=VOCAB_SIZE,
    n_text_ctx=MAX_TOKENS,
    n_state=CONFIG.model.state,
    n_head=CONFIG.model.head,
    n_layer=CONFIG.model.layer,
)
model = Whisper(dims).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────── 2. 数据 ───────────────────────

print("Loading dataset...")
train_full_ds = ChartDataset(
    CHARTS_DIR,
    cache_dir=CONFIG.paths.mert_cache_dir,
    level_idx=LEVEL_IDX,
    max_tokens=MAX_TOKENS,
    stride_sec=CONFIG.window.train_stride_sec,
    mert_frames=dims.n_audio_ctx,
    chart_limit=OVERFIT_CHARTS,
    seed=SEED,
)
val_full_ds = ChartDataset(
    CHARTS_DIR,
    cache_dir=CONFIG.paths.mert_cache_dir,
    level_idx=LEVEL_IDX,
    max_tokens=MAX_TOKENS,
    stride_sec=CONFIG.window.infer_stride_sec,
    mert_frames=dims.n_audio_ctx,
    valid_pairs=train_full_ds.valid_pairs,
)
print(f"Total train windows: {len(train_full_ds)}, val windows: {len(val_full_ds)}")

by_chart = defaultdict(list)
for idx, entry in enumerate(train_full_ds._index):
    by_chart[entry.mel_path].append(idx)

chart_paths = sorted(by_chart)
perm = torch.randperm(len(chart_paths), generator=torch.Generator().manual_seed(SEED)).tolist()
if OVERFIT_CHARTS > 0:
    selected_charts = {chart_paths[i] for i in perm[:min(OVERFIT_CHARTS, len(chart_paths))]}
    val_charts = selected_charts
    train_chart_count = len(selected_charts)
    val_chart_count = len(selected_charts)
    train_indices = [i for p in chart_paths if p in selected_charts for i in by_chart[p]]
    val_indices = [i for i, entry in enumerate(val_full_ds._index) if entry.mel_path in selected_charts]
else:
    val_chart_count = max(1, int(len(chart_paths) * VAL_RATIO))
    train_chart_count = len(chart_paths) - val_chart_count
    val_charts = {chart_paths[i] for i in perm[:val_chart_count]}
    train_indices = [i for p in chart_paths if p not in val_charts for i in by_chart[p]]
    val_indices = [i for i, entry in enumerate(val_full_ds._index) if entry.mel_path in val_charts]
base_train_ds = Subset(train_full_ds, train_indices)
augmented_train_ds = AudioAugmentedDataset(
    base_train_ds,
    probability=CONFIG.training.audio_augment_probability,
    noise_std=CONFIG.training.feature_noise_std,
    max_time_mask_frames=CONFIG.training.max_time_mask_frames,
)
train_ds = RotatedDataset(augmented_train_ds, rotations=CONFIG.training.rotations)
val_ds = Subset(val_full_ds, val_indices)
if len(base_train_ds) == 0:
    raise ValueError("训练集没有可用窗口；正常训练至少需要两首含指定难度的有效歌曲")
if len(val_ds) == 0:
    raise ValueError("验证集没有可用窗口；请检查谱面目录、难度编号和验证集比例")
print(
    f"Train: {len(base_train_ds)} windows / {train_chart_count} charts, "
    f"旋转增强后 {len(train_ds)} samples, "
    f"Val: {len(val_ds)} windows / {val_chart_count} charts"
)
supervised_tokens = 0
eos_targets = 0
after_frame_end = 0
for index in train_indices:
    entry = train_full_ds._index[index]
    supervised_tokens += sum(entry.loss_mask)
    eos_targets += int(entry.tokens[-1] == EOS and entry.loss_mask[-1])
    after_frame_end += sum(
        entry.loss_mask[i] and entry.tokens[i - 1] == FRAME_END
        for i in range(1, len(entry.tokens))
    )
print(
    f"EOS监督: {eos_targets}/{supervised_tokens} token ({eos_targets / max(supervised_tokens, 1) * 100:.3f}%), "
    f"FRAME_END后结束率={eos_targets / max(after_frame_end, 1) * 100:.2f}%, "
    f"EOS loss权重={EOS_LOSS_WEIGHT:g}x"
)
if OVERFIT_CHARTS > 0:
    print(f"整曲过拟合模式: {train_chart_count} 首歌，训练和验证使用同一批歌曲")

chart_by_mel = {mel_path: chart_path for chart_path, mel_path in val_full_ds.valid_pairs}
if OVERFIT_CHARTS > 0:
    print("过拟合歌曲:")
    for i, mel_path in enumerate(sorted(selected_charts), 1):
        chart_dir = chart_by_mel[mel_path].parent.relative_to(CHARTS_DIR)
        print(f"  {i}. {chart_dir}")
    gen_val_paths = sorted(selected_charts)
else:
    gen_val_paths = sorted(val_charts)[:VAL_GEN_CHARTS]

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=partial(collate_segments, mert_frames=dims.n_audio_ctx),
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=partial(collate_segments, mert_frames=dims.n_audio_ctx),
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
)

# ─────────────────────── 3. 损失 & 优化器 ───────────────────────

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD,
    label_smoothing=LABEL_SMOOTHING,
    reduction="none",
)
optimizer = optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    fused=DEVICE.type == "cuda",
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
window_config = {
    "mert_frames": dims.n_audio_ctx,
    "prefix_start_sec": CONFIG.window.prefix_start_sec,
    "target_start_sec": CONFIG.window.target_start_sec,
    "target_end_sec": CONFIG.window.target_end_sec,
    "train_stride_sec": CONFIG.window.train_stride_sec,
    "infer_stride_sec": CONFIG.window.infer_stride_sec,
}
start_epoch = 1
resume_checkpoint = None
if RESUME_PATH:
    resume_path = Path(RESUME_PATH)
    if not resume_path.is_file():
        raise ValueError(f"恢复检查点不存在或不是文件: {resume_path}")
    resume_checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    if resume_checkpoint.get("checkpoint_version") != 3 or resume_checkpoint.get("model_kind") != "mert-window-decoder-v2":
        raise ValueError("检查点来自旧架构，不能继续训练")
    if resume_checkpoint.get("feature_signature") != feature_signature():
        raise ValueError("检查点使用的 MERT 特征实现与当前环境不一致")
    validate_checkpoint_config(
        resume_checkpoint.get("config"),
        CONFIG,
        for_training=True,
    )
    resume_window_config = resume_checkpoint.get("window_config")
    if resume_window_config is None:
        print("警告: 旧 checkpoint 没有 window_config，无法确认是否来自当前滑窗任务")
    elif resume_window_config != window_config:
        raise ValueError(
            f"checkpoint 窗口配置不兼容: {resume_window_config!r}，当前为 {window_config!r}"
        )
    model.load_state_dict(resume_checkpoint["model_state_dict"])
    if "optimizer_state_dict" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_checkpoint:
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    else:
        # 旧 checkpoint 没有 scheduler 状态；optimizer 中已经保存了当时的学习率。
        # 直接恢复计数，避免在 optimizer.step() 前伪调用 scheduler.step()。
        completed_epochs = int(resume_checkpoint.get("epoch", 0))
        scheduler.last_epoch = completed_epochs
        scheduler._step_count = completed_epochs + 1
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    if "scaler_state_dict" in resume_checkpoint:
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
    start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
    print(f"从检查点继续训练: {resume_path}，下一轮 epoch={start_epoch}")

# ─────────────────────── 4. 训练/验证函数 ───────────────────────


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    progress = tqdm(loader, desc="训练", leave=False)

    for step, batch in enumerate(progress, 1):
        mel = batch["mel"].to(device, non_blocking=True)       # (B, MERT帧, 状态维度)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)  # (B, S)
        mask = batch["mask"].to(device, non_blocking=True)      # (B, S)  True=有效
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        # teacher forcing: tokens 已经以 SOS 开头，只预测下一个 token。
        dec_input = tokens[:, :-1]
        target = tokens[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(mel, dec_input, audio_mask)      # (B, S, vocab)
            token_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            ).view_as(target)
            target_loss_mask = loss_mask[:, 1:]
            token_loss = token_loss * torch.where(target == EOS, EOS_LOSS_WEIGHT, 1.0)
            loss = token_loss[target_loss_mask].mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        n_valid = target_loss_mask.sum()
        total_loss += loss.detach() * n_valid
        total_tokens += n_valid
        if step % 10 == 0:
            progress.set_postfix(loss=f"{(total_loss / total_tokens.clamp_min(1)).item():.4f}")

    return (total_loss / total_tokens.clamp_min(1)).item()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    eos_correct = torch.zeros((), device=device)
    eos_total = torch.zeros((), device=device)
    eos_probability = torch.zeros((), device=device)
    eos_rank = torch.zeros((), device=device)
    eos_as_frame_start = torch.zeros((), device=device)

    progress = tqdm(loader, desc="验证", leave=False)
    for step, batch in enumerate(progress, 1):
        mel = batch["mel"].to(device, non_blocking=True)
        audio_mask = batch["audio_mask"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        dec_input = tokens[:, :-1]
        target = tokens[:, 1:]
        target_mask = loss_mask[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(mel, dec_input, audio_mask)
            token_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            ).view_as(target)
            token_loss = token_loss * torch.where(target == EOS, EOS_LOSS_WEIGHT, 1.0)
            loss = token_loss[target_mask].mean()

        n_valid = target_mask.sum()
        total_loss += loss * n_valid
        total_tokens += n_valid

        preds = logits.argmax(dim=-1)
        # mask 排除 PAD 位置，只统计有效 token 的准确率
        correct += ((preds == target) & target_mask).sum()
        eos_mask = (target == EOS) & target_mask
        if eos_mask.any():
            eos_logits = logits[eos_mask]
            eos_targets = target[eos_mask]
            eos_total += eos_mask.sum()
            eos_correct += (eos_logits.argmax(dim=-1) == eos_targets).sum()
            eos_probability += eos_logits.softmax(dim=-1)[:, EOS].sum()
            eos_rank += (eos_logits > eos_logits[:, EOS].unsqueeze(1)).sum()
            eos_as_frame_start += (eos_logits.argmax(dim=-1) == FRAME_START).sum()

        if step % 10 == 0:
            progress.set_postfix(loss=f"{(total_loss / total_tokens.clamp_min(1)).item():.4f}")

    avg_loss = total_loss / total_tokens.clamp_min(1)
    accuracy = correct / total_tokens.clamp_min(1) * 100
    return (
        avg_loss.item(),
        accuracy.item(),
        (eos_correct / eos_total.clamp_min(1) * 100).item(),
        (eos_probability / eos_total.clamp_min(1) * 100).item(),
        (eos_rank / eos_total.clamp_min(1) + 1).item(),
        (eos_as_frame_start / eos_total.clamp_min(1) * 100).item(),
    )


@torch.no_grad()
def validate_generation(model, mel_paths, chart_by_mel, level_idx, device):
    model.eval()
    total_tp = 0
    total_pred = 0
    total_target = 0
    total_windows = 0
    total_limit_windows = 0
    total_dead_end_windows = 0
    total_eos_windows = 0
    total_new_tokens = 0
    total_raw_new_tokens = 0
    total_dropped_tokens = 0
    for mel_path in tqdm(mel_paths, desc="整曲生成验证", leave=False):
        mel = np.load(mel_path)
        generated_frames, infer_stats = overlap_infer(model, mel, device, verbose=False, return_stats=True)
        total_windows += infer_stats["windows"]
        total_limit_windows += infer_stats["limit_windows"]
        total_dead_end_windows += infer_stats["dead_end_windows"]
        total_eos_windows += infer_stats["eos_windows"]
        total_new_tokens += infer_stats["new_tokens"]
        total_raw_new_tokens += infer_stats["raw_new_tokens"]
        total_dropped_tokens += infer_stats["dropped_tokens"]
        c = compiler()
        c.parse(chart_by_mel[mel_path].read_text(encoding="utf-8"))
        level = c.chart.all_levels[level_idx]
        target_frames = [] if level is None else level.frames
        tp, pred_count, target_count = content_match_frame_counts(generated_frames, target_frames)
        total_tp += tp
        total_pred += pred_count
        total_target += target_count
    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_target, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (
        precision * 100,
        recall * 100,
        f1 * 100,
        total_limit_windows,
        total_dead_end_windows,
        total_eos_windows,
        total_windows,
        total_raw_new_tokens / max(total_windows, 1),
        total_new_tokens / max(total_windows, 1),
        total_dropped_tokens / max(total_windows, 1),
    )


@torch.no_grad()
def validate_oracle_prefix(model, loader, device, max_windows):
    model.eval()
    checked = 0
    limit_windows = 0
    dead_end_windows = 0
    total_new_tokens = 0
    total_raw_new_tokens = 0
    total_dropped_tokens = 0
    for batch in loader:
        for i in range(batch["tokens"].size(0)):
            if checked >= max_windows:
                return (
                    limit_windows,
                    dead_end_windows,
                    checked,
                    total_raw_new_tokens / max(checked, 1),
                    total_new_tokens / max(checked, 1),
                    total_dropped_tokens / max(checked, 1),
                )
            valid_len = int(batch["mask"][i].sum().item())
            loss_positions = batch["loss_mask"][i, :valid_len].nonzero()
            first_target = int(loss_positions[0].item())
            prefix = batch["tokens"][i, :first_target].tolist()
            _tokens, stats = decode_segment(
                model,
                batch["mel"][i],
                device,
                max_tokens=MAX_TOKENS,
                prefix_tokens=prefix,
                audio_mask=batch["audio_mask"][i],
                return_stats=True,
                verbose=False,
            )
            checked += 1
            limit_windows += int(stats["hit_limit"])
            dead_end_windows += int(stats["grammar_dead_end"])
            total_new_tokens += int(stats["new_tokens"])
            total_raw_new_tokens += int(stats["raw_new_tokens"])
            total_dropped_tokens += int(stats["dropped_tokens"])
    return (
        limit_windows,
        dead_end_windows,
        checked,
        total_raw_new_tokens / max(checked, 1),
        total_new_tokens / max(checked, 1),
        total_dropped_tokens / max(checked, 1),
    )


# ─────────────────────── 5. 主循环 ───────────────────────

if __name__ == "__main__":
    best_val_loss = float("inf")
    best_val_content_f1 = -1.0
    bad_epochs = 0

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_loss, val_acc, eos_acc, eos_prob, eos_rank, eos_to_frame = validate(
            model, val_loader, criterion, DEVICE
        )
        oracle_limits, oracle_dead_ends, oracle_windows, oracle_avg_raw_tokens, oracle_avg_tokens, oracle_avg_dropped = validate_oracle_prefix(
            model, val_loader, DEVICE, VAL_ORACLE_WINDOWS
        )
        (
            val_precision,
            val_recall,
            val_content_f1,
            limit_windows,
            dead_end_windows,
            eos_windows,
            gen_windows,
            avg_raw_tokens,
            avg_new_tokens,
            avg_dropped_tokens,
        ) = validate_generation(
            model, gen_val_paths, chart_by_mel, LEVEL_IDX, DEVICE
        )
        scheduler.step()
        val_loss_improved = val_loss < best_val_loss

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}]  "
            f"学习率={lr_now:.2e}  "
            f"训练损失={train_loss:.4f}  "
            f"验证损失={val_loss:.4f}  "
            f"看答案续写猜对率={val_acc:.2f}%  "
            f"EOS准确率={eos_acc:.2f}%  "
            f"EOS概率={eos_prob:.2f}%  "
            f"EOS平均排名={eos_rank:.1f}  "
            f"EOS误判FRAME_START={eos_to_frame:.2f}%  "
            f"真值前缀撞上限={oracle_limits}/{oracle_windows}  "
            f"真值前缀约束死路={oracle_dead_ends}/{oracle_windows}  "
            f"真值前缀原始/保留/丢弃token={oracle_avg_raw_tokens:.1f}/{oracle_avg_tokens:.1f}/{oracle_avg_dropped:.1f}  "
            f"整曲生成P={val_precision:.2f}%  "
            f"整曲生成R={val_recall:.2f}%  "
            f"整曲生成F1={val_content_f1:.2f}%  "
            f"撞上限={limit_windows}/{gen_windows}  "
            f"约束死路={dead_end_windows}/{gen_windows}  "
            f"正常结束={eos_windows}/{gen_windows}  "
            f"原始/保留/丢弃token={avg_raw_tokens:.1f}/{avg_new_tokens:.1f}/{avg_dropped_tokens:.1f}"
        )

        if val_loss_improved:
            best_val_loss = val_loss
            bad_epochs = 0
            ckpt_path = CHECKPOINT_DIR / "best.pt"
            torch.save({
                "checkpoint_version": 3,
                "model_kind": "mert-window-decoder-v2",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_content_f1": val_content_f1,
                "dims": dims,
                "window_config": window_config,
                "config": checkpoint_config(CONFIG),
                "feature_signature": feature_signature(),
            }, ckpt_path)
            print(f"  -> 保存验证损失最优模型到 {ckpt_path}")
        if val_content_f1 > best_val_content_f1:
            best_val_content_f1 = val_content_f1
            ckpt_path = CHECKPOINT_DIR / "best_gen.pt"
            torch.save({
                "checkpoint_version": 3,
                "model_kind": "mert-window-decoder-v2",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_content_f1": val_content_f1,
                "dims": dims,
                "window_config": window_config,
                "config": checkpoint_config(CONFIG),
                "feature_signature": feature_signature(),
            }, ckpt_path)
            print(f"  -> 保存生成效果最优模型到 {ckpt_path}")
        ckpt_path = CHECKPOINT_DIR / "newest.pt"
        torch.save({
            "checkpoint_version": 3,
            "model_kind": "mert-window-decoder-v2",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_content_f1": val_content_f1,
            "dims": dims,
            "window_config": window_config,
            "config": checkpoint_config(CONFIG),
            "feature_signature": feature_signature(),
        }, ckpt_path)
        if not val_loss_improved:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"  -> 验证集连续 {EARLY_STOP_PATIENCE} 轮没有改善，提前停止")
                break
