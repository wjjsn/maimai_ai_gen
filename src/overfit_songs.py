"""10 首歌曲小规模学习诊断。"""

import numpy as np
import torch

from config import CONFIG
from dataset import (
    _stack_batch, _window, build_dataset_index, load_song_data, split_songs, window_starts,
)
from infer import model_dimensions
from model import NoteTimingTransformer
from train import DEVICE, evaluate_loaded_songs, run_epoch


TRAIN_SONGS = 10
VALIDATION_SONGS = 2
MAX_EPOCHS = 60
EVALUATE_EVERY = 5


def _metrics(model, loaded):
    result = evaluate_loaded_songs(model, loaded)["tolerances"]
    return result["3"]["total"]["macro_f1"], result["6"]["total"]["macro_f1"]


def _batches(loaded, epoch: int):
    descriptors = [
        (features, level, tracks, start)
        for _song, features, levels in loaded
        for level, tracks in levels
        for start in window_starts(len(features), 0, False)
    ]
    rng = np.random.default_rng(CONFIG.training.seed + epoch)
    rng.shuffle(descriptors)
    batch = []
    for features, level, tracks, start in descriptors:
        sample = _window(features, tracks, start)
        sample["difficulty"] = torch.tensor(level.difficulty, dtype=torch.float32)
        batch.append(sample)
        if len(batch) == CONFIG.training.batch_size:
            yield _stack_batch(batch)
            batch.clear()
    if batch:
        yield _stack_batch(batch)


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    np.random.seed(CONFIG.training.seed)
    splits = split_songs(build_dataset_index())
    train_songs = splits["train"][:TRAIN_SONGS]
    validation_songs = splits["validation"][:VALIDATION_SONGS]
    loaded_train = tuple((song, *load_song_data(song)) for song in train_songs)
    loaded_validation = tuple((song, *load_song_data(song)) for song in validation_songs)
    model = NoteTimingTransformer(model_dimensions()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    print(
        f"设备={DEVICE} 训练歌曲={len(train_songs)} 验证歌曲={len(validation_songs)} "
        f"训练难度={sum(len(song.levels) for song in train_songs)}",
        flush=True,
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        losses, updates = run_epoch(
            model, _batches(loaded_train, epoch), optimizer, scheduler, scaler,
            f"10首训练 epoch={epoch}",
        )
        if epoch % EVALUATE_EVERY:
            print(f"epoch={epoch} updates={updates} loss={losses[0]:.6f}", flush=True)
            continue
        train_f1_3, train_f1_6 = _metrics(model, loaded_train)
        val_f1_3, val_f1_6 = _metrics(model, loaded_validation)
        print(
            f"epoch={epoch} updates={updates} loss={losses[0]:.6f} "
            f"训练F1@3={train_f1_3:.2%} 训练F1@6={train_f1_6:.2%} "
            f"验证F1@3={val_f1_3:.2%} 验证F1@6={val_f1_6:.2%}",
            flush=True,
        )
        if train_f1_6 >= 0.95:
            print("10 首训练集整曲过拟合成功", flush=True)
            return
    print("10 首训练集未达到 95% F1@6", flush=True)


if __name__ == "__main__":
    main()
