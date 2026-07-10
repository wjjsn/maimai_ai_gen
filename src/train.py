import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from model import Whisper, ModelDimensions
from dataset import ChartDataset, collate_segments
from maidata_parser import (
    PAD, SOS, EOS, VOCAB_SIZE,
    FRAME_START, FRAME_END, SEGMENT_START, SEGMENT_END,
)

# ─────────────────────── 0. 配置 ───────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
LEVEL_IDX = 4          # 默认 Master
MAX_TOKENS = 2048      # 序列最大长度
BATCH_SIZE = 16
NUM_EPOCHS = 500
LR = 3e-4
WEIGHT_DECAY = 0.01
VAL_RATIO = 0.15       # 验证集比例
GRAD_CLIP = 1.0        # 梯度裁剪
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ─────────────────────── 1. 模型 ───────────────────────

dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=VOCAB_SIZE,
    n_text_ctx=MAX_TOKENS,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)
model = Whisper(dims).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────── 2. 数据 ───────────────────────

print("Loading dataset...")
full_ds = ChartDataset(CHARTS_DIR, level_idx=LEVEL_IDX, max_tokens=MAX_TOKENS)
print(f"Total segments: {len(full_ds)}")

val_size = max(1, int(len(full_ds) * VAL_RATIO))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
print(f"Train: {train_size}, Val: {val_size}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=partial(collate_segments, mel_frames=2 * dims.n_audio_ctx),
    num_workers=6, pin_memory=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=partial(collate_segments, mel_frames=2 * dims.n_audio_ctx),
    num_workers=6, pin_memory=True,
)

# ─────────────────────── 3. 损失 & 优化器 ───────────────────────

# 给特殊标记更高权重，帮助模型学会 EOS 停止和结构边界
# 注意: weight 按类别加权，ignore_index 按位置排除，两者职责不同
# PAD 只出现在 EOS 之后的位置，被 ignore_index 排除，不需要设置权重
_token_weights = torch.ones(VOCAB_SIZE)
_token_weights[EOS] = 5.0           # 让模型学会及时停止
_token_weights[SOS] = 3.0           # 强化起始信号
_token_weights[FRAME_START] = 3.0   # 帧边界
_token_weights[FRAME_END] = 3.0     # 帧边界
_token_weights[SEGMENT_START] = 3.0 # 段边界
_token_weights[SEGMENT_END] = 3.0   # 段边界

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD,
    weight=_token_weights.to(DEVICE),
)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ─────────────────────── 4. 训练/验证函数 ───────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        mel = batch["mel"].to(device)       # (B, n_mels, T_mel)
        tokens = batch["tokens"].to(device)  # (B, S)
        mask = batch["mask"].to(device)      # (B, S)  True=有效

        # teacher forcing: tokens 已经以 SOS 开头，只预测下一个 token。
        dec_input = tokens[:, :-1]
        target = tokens[:, 1:]

        logits = model(mel, dec_input)      # (B, S, vocab)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        n_valid = mask[:, 1:].sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    for batch in loader:
        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        dec_input = tokens[:, :-1]
        target = tokens[:, 1:]
        target_mask = mask[:, 1:]

        logits = model(mel, dec_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )

        n_valid = target_mask.sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

        preds = logits.argmax(dim=-1)
        # mask 排除 PAD 位置，只统计有效 token 的准确率
        correct += ((preds == target) & target_mask).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct / max(total_tokens, 1) * 100
    return avg_loss, accuracy


# ─────────────────────── 5. 主循环 ───────────────────────

if __name__ == "__main__":
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}]  "
            f"lr={lr_now:.2e}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CHECKPOINT_DIR / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "dims": dims,
            }, ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}")
