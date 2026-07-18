"""固定单窗口过拟合诊断。"""

import numpy as np
import torch

from config import CONFIG
from dataset import (
    HOLD_START_COUNT, TAP_COUNT, WINDOW_FRAMES, _window, build_dataset_index,
    load_song_data, split_songs,
)
from infer import infer_features, model_dimensions
from model import NoteTimingTransformer
from train import DEVICE, compute_loss, evaluate_tracks


MAX_STEPS = 2000
EVALUATE_EVERY = 25


def _sample():
    songs = split_songs(build_dataset_index())["train"]
    record = songs[0]
    features, levels = load_song_data(record)
    level, tracks = levels[-1]
    event_frames = np.flatnonzero(tracks[:, TAP_COUNT] | tracks[:, HOLD_START_COUNT])
    if not event_frames.size:
        raise ValueError("诊断歌曲没有事件")
    center = int(event_frames[len(event_frames) // 2])
    start = max(0, min(len(features) - WINDOW_FRAMES, center - WINDOW_FRAMES // 2))
    sample = _window(features, tracks, start)
    sample["difficulty"] = torch.tensor(level.difficulty, dtype=torch.float32)
    batch = {key: value.unsqueeze(0).to(DEVICE) for key, value in sample.items()}
    return record, level, features[start:start + WINDOW_FRAMES], tracks[start:start + WINDOW_FRAMES], batch


def main() -> None:
    torch.manual_seed(CONFIG.training.seed)
    record, level, features, target, batch = _sample()
    model = NoteTimingTransformer(model_dimensions()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    print(
        f"设备={DEVICE} 固定窗口歌曲={record.title} 难度={level.difficulty} "
        f"事件={int(target[:, :2].sum())}",
        flush=True,
    )
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(batch["features"], batch["difficulty"], batch["mask"]), batch)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.grad_clip)
        optimizer.step()
        if step % EVALUATE_EVERY:
            continue
        model.eval()
        predicted, _ = infer_features(model, features, level.difficulty, DEVICE)
        strict = evaluate_tracks(target, predicted, 3)["total"]["f1"]
        main = evaluate_tracks(target, predicted, 6)["total"]["f1"]
        print(f"step={step} loss={loss.item():.6f} F1@3={strict:.2%} F1@6={main:.2%}", flush=True)
        if main >= 0.999:
            print("固定窗口过拟合成功", flush=True)
            return
    raise RuntimeError("固定窗口在最大步数内未能过拟合")


if __name__ == "__main__":
    main()
