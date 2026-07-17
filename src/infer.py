"""Transformer 逐帧事件模型的正式整曲推理入口。"""

from pathlib import Path
import shutil

import numpy as np
import torch

from config import CONFIG, checkpoint_config
from dataset import (
    HOLD_DURATION_1, HOLD_DURATION_2, HOLD_START_COUNT, TAP_COUNT, TRACK_COUNT, WINDOW_FRAMES,
    extract_log_mel, tracks_to_maidata,
)
from model import ModelDimensions, NoteTimingTransformer


MODEL_KIND = "log-mel-difficulty-bert-window-event-v3"


def model_dimensions() -> ModelDimensions:
    return ModelDimensions(
        CONFIG.audio.n_mels, CONFIG.model.hidden_dim, CONFIG.model.layers,
        CONFIG.model.dropout, WINDOW_FRAMES, CONFIG.model.attention_heads,
    )


def load_model(path: str | Path, device: torch.device) -> NoteTimingTransformer:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("checkpoint_version") != 3 or state.get("model_kind") != MODEL_KIND:
        raise ValueError("检查点来自不兼容架构")
    if state.get("config") != checkpoint_config() or state.get("dims") != model_dimensions():
        raise ValueError("检查点配置与当前配置不一致")
    model = NoteTimingTransformer(model_dimensions()).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[infer] 加载检查点 epoch={state.get('epoch', '?')}")
    return model


def _starts(length: int, window: int = WINDOW_FRAMES, stride: int | None = None) -> list[int]:
    stride = stride or window // 2
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.no_grad()
def predict_tracks(
    model: NoteTimingTransformer, features: np.ndarray, difficulty: float,
    device: torch.device,
) -> np.ndarray:
    length = len(features)
    tap_sum = np.zeros((length, 3), dtype=np.float32)
    hold_sum = np.zeros((length, 3), dtype=np.float32)
    duration_sum = np.zeros((length, 2), dtype=np.float32)
    weight_sum = np.zeros(length, dtype=np.float32)
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    window_weight = np.hanning(WINDOW_FRAMES + 2)[1:-1].astype(np.float32)
    window_weight = np.maximum(window_weight, 1e-3)
    for start in _starts(length):
        end = min(start + WINDOW_FRAMES, length)
        valid = end - start
        window = np.zeros((WINDOW_FRAMES, features.shape[1]), dtype=np.float32)
        window[:valid] = features[start:end]
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        mask = torch.zeros(1, WINDOW_FRAMES, dtype=torch.bool, device=device)
        mask[:, :valid] = True
        level = torch.tensor([difficulty], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tap, hold, duration = model(tensor, level, mask)
        weight = window_weight[:valid, None]
        tap_sum[start:end] += tap[0, :valid].float().cpu().numpy() * weight
        hold_sum[start:end] += hold[0, :valid].float().cpu().numpy() * weight
        duration_sum[start:end] += duration[0, :valid].float().cpu().numpy() * weight
        weight_sum[start:end] += window_weight[:valid]
    tap = (tap_sum / weight_sum[:, None]).argmax(axis=-1)
    hold = (hold_sum / weight_sum[:, None]).argmax(axis=-1)
    normalized_duration = duration_sum / weight_sum[:, None]
    durations = np.rint(np.expm1(normalized_duration * np.log1p(maximum))).astype(np.int32)
    tracks = np.column_stack((tap, hold, durations)).astype(np.int32)
    tracks[hold == 0, HOLD_DURATION_1:] = 0
    tracks[hold == 1, HOLD_DURATION_1 + 1] = 0
    return tracks


def filter_tracks(tracks: np.ndarray) -> tuple[np.ndarray, int]:
    tracks = tracks.copy()
    dropped = 0
    last_tap = -CONFIG.inference.short_min_gap_frames
    for frame in np.flatnonzero(tracks[:, TAP_COUNT]):
        if frame - last_tap < CONFIG.inference.short_min_gap_frames:
            dropped += int(tracks[frame, TAP_COUNT])
            tracks[frame, TAP_COUNT] = 0
        else:
            last_tap = int(frame)
    minimum = max(
        CONFIG.inference.long_min_frames,
        round(CONFIG.inference.min_duration_sec * CONFIG.audio.frames_per_sec),
    )
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    for frame in np.flatnonzero(tracks[:, HOLD_START_COUNT]):
        count = int(tracks[frame, HOLD_START_COUNT])
        durations = np.clip(tracks[frame, HOLD_DURATION_1:HOLD_DURATION_1 + count], minimum, maximum)
        tracks[frame, HOLD_DURATION_1:HOLD_DURATION_1 + count] = durations
        tracks[frame, HOLD_DURATION_1 + count:] = 0
    tracks[tracks[:, HOLD_START_COUNT] == 0, HOLD_DURATION_1:] = 0
    return tracks, dropped


def infer_features(
    model: NoteTimingTransformer, features: np.ndarray, difficulty: float,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    return filter_tracks(predict_tracks(model, features, difficulty, device))


def save_inference(audio_path: Path, text: str, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(audio_path, output_dir / "track.mp3")
    path = output_dir / "maidata.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _self_check() -> None:
    assert _starts(100) == [0]
    assert _starts(1200) == [0, 176]
    tracks = np.zeros((20, TRACK_COUNT), dtype=np.int32)
    tracks[3] = (2, 1, 1, 8)
    tracks[5, TAP_COUNT] = 1
    filtered, dropped = filter_tracks(tracks)
    assert dropped == 1 and filtered[3, HOLD_DURATION_1] >= CONFIG.inference.long_min_frames
    assert filtered[3, HOLD_DURATION_2] == 0
    print("[infer] 自检通过")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_path = CONFIG.inference.audio_path
    model = load_model(CONFIG.inference.checkpoint, device)
    tracks, dropped = infer_features(
        model, extract_log_mel(audio_path), CONFIG.inference.level_query, device,
    )
    text = tracks_to_maidata(
        tracks, CONFIG.inference.level_idx, audio_path.stem, CONFIG.inference.level_query,
    )
    output_dir = CONFIG.paths.inference_output_dir / audio_path.stem
    path = save_inference(audio_path, text, output_dir)
    print(f"[infer] 音符={int(tracks[:, :2].sum())} 丢弃短音={dropped} 输出={path}")


if __name__ == "__main__":
    main()
