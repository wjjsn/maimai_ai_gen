"""Transformer 逐帧事件模型的正式整曲推理入口。"""

from pathlib import Path
import shutil

import numpy as np
import torch

from config import CONFIG, checkpoint_config
from dataset import (
    HOLD_DURATION_1, HOLD_DURATION_2, HOLD_START_COUNT, TAP_COUNT, TRACK_COUNT, WINDOW_FRAMES,
    extract_log_mel, tracks_to_maidata, window_starts,
)
from model import ModelDimensions, NoteTimingTransformer


MODEL_KIND = "log-mel-difficulty-bert-window-event-v3"
LABEL_PROTOCOL = "event-heatmap-balanced-shuffled-v5"


def model_dimensions() -> ModelDimensions:
    return ModelDimensions(
        CONFIG.audio.n_mels, CONFIG.model.hidden_dim, CONFIG.model.layers,
        CONFIG.model.dropout, WINDOW_FRAMES, CONFIG.model.attention_heads,
    )


def load_model(path: str | Path, device: torch.device) -> NoteTimingTransformer:
    state = torch.load(path, map_location=device, weights_only=False)
    if (
        state.get("checkpoint_version") != 3
        or state.get("model_kind") != MODEL_KIND
        or state.get("label_protocol") != LABEL_PROTOCOL
    ):
        raise ValueError("检查点来自不兼容架构")
    saved_config = state.get("config", {})
    if (
        state.get("dims") != model_dimensions()
        or any(saved_config.get(key) != value for key, value in checkpoint_config().items())
    ):
        raise ValueError("检查点配置与当前配置不一致")
    model = NoteTimingTransformer(model_dimensions()).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[infer] 加载检查点 epoch={state.get('epoch', '?')}")
    return model


@torch.no_grad()
def predict_tracks(
    model: NoteTimingTransformer, features: np.ndarray, difficulty: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = len(features)
    tap_sum = np.zeros((length, 3), dtype=np.float32)
    hold_sum = np.zeros((length, 3), dtype=np.float32)
    duration_sum = np.zeros((length, 2), dtype=np.float32)
    weight_sum = np.zeros(length, dtype=np.float32)
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    window_weight = np.hanning(WINDOW_FRAMES + 2)[1:-1].astype(np.float32)
    window_weight = np.maximum(window_weight, 1e-3)
    for start in window_starts(length, 0, False):
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
    tap_logits = tap_sum / weight_sum[:, None]
    hold_logits = hold_sum / weight_sum[:, None]
    tap_score = tap_logits[:, 1:].max(axis=-1) - tap_logits[:, 0]
    hold_score = hold_logits[:, 1:].max(axis=-1) - hold_logits[:, 0]
    tap = np.where(tap_score >= 0, tap_logits[:, 1:].argmax(axis=-1) + 1, 0)
    hold = np.where(hold_score >= 0, hold_logits[:, 1:].argmax(axis=-1) + 1, 0)
    normalized_duration = duration_sum / weight_sum[:, None]
    durations = np.rint(np.expm1(normalized_duration * np.log1p(maximum))).astype(np.int32)
    tracks = np.column_stack((tap, hold, durations)).astype(np.int32)
    tracks[hold == 0, HOLD_DURATION_1:] = 0
    tracks[hold == 1, HOLD_DURATION_1 + 1] = 0
    return tracks, tap_score, hold_score


def _keep_local_peaks(
    tracks: np.ndarray, column: int, scores: np.ndarray, minimum_gap: int,
) -> int:
    frames = np.flatnonzero(tracks[:, column])
    remaining = set(int(frame) for frame in frames)
    dropped = 0
    for keep in sorted(remaining, key=lambda frame: (-float(scores[frame]), frame)):
        if keep not in remaining:
            continue
        remaining.remove(keep)
        nearby = [frame for frame in remaining if abs(frame - keep) < minimum_gap]
        for frame in nearby:
            dropped += int(tracks[frame, column])
            tracks[frame, column] = 0
            remaining.remove(frame)
    return dropped


def filter_tracks(
    tracks: np.ndarray, tap_scores: np.ndarray | None = None,
    hold_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    tracks = tracks.copy()
    tap_scores = tap_scores if tap_scores is not None else tracks[:, TAP_COUNT]
    hold_scores = hold_scores if hold_scores is not None else tracks[:, HOLD_START_COUNT]
    gap = max(1, CONFIG.inference.short_min_gap_frames)
    dropped = _keep_local_peaks(tracks, TAP_COUNT, tap_scores, gap)
    dropped += _keep_local_peaks(tracks, HOLD_START_COUNT, hold_scores, gap)
    minimum = max(
        CONFIG.inference.long_min_frames,
        round(CONFIG.inference.min_duration_sec * CONFIG.audio.frames_per_sec),
    )
    maximum = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    for frame in np.flatnonzero(tracks[:, HOLD_START_COUNT]):
        count = int(tracks[frame, HOLD_START_COUNT])
        remaining = len(tracks) - frame
        upper = min(maximum, remaining)
        durations = np.clip(
            tracks[frame, HOLD_DURATION_1:HOLD_DURATION_1 + count],
            min(minimum, upper), upper,
        )
        tracks[frame, HOLD_DURATION_1:HOLD_DURATION_1 + count] = durations
        tracks[frame, HOLD_DURATION_1 + count:] = 0
    tracks[tracks[:, HOLD_START_COUNT] == 0, HOLD_DURATION_1:] = 0
    return tracks, dropped


def infer_features(
    model: NoteTimingTransformer, features: np.ndarray, difficulty: float,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    tracks, tap_scores, hold_scores = predict_tracks(model, features, difficulty, device)
    return filter_tracks(tracks, tap_scores, hold_scores)


def save_inference(audio_path: Path, text: str, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(audio_path, output_dir / "track.mp3")
    path = output_dir / "maidata.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _self_check() -> None:
    tracks = np.zeros((20, TRACK_COUNT), dtype=np.int32)
    tracks[3] = (2, 1, 1, 8)
    tracks[5, TAP_COUNT] = 1
    tracks[6, HOLD_START_COUNT:] = (1, 12, 0)
    tap_scores = np.zeros(20, dtype=np.float32)
    tap_scores[3], tap_scores[5] = 0.2, 0.8
    hold_scores = np.zeros(20, dtype=np.float32)
    hold_scores[3], hold_scores[6] = 0.9, 0.1
    filtered, dropped = filter_tracks(tracks, tap_scores, hold_scores)
    assert dropped == 3 and filtered[5, TAP_COUNT] == 1 and filtered[3, TAP_COUNT] == 0
    assert filtered[3, HOLD_DURATION_1] >= CONFIG.inference.long_min_frames
    assert filtered[6, HOLD_START_COUNT] == 0
    assert filtered[3, HOLD_DURATION_2] == 0
    tail = np.zeros((12, TRACK_COUNT), dtype=np.int32)
    tail[1, HOLD_START_COUNT:] = (1, 100, 0)
    tail, _ = filter_tracks(tail)
    assert tail[1, HOLD_DURATION_1] == 11
    tail[11, HOLD_START_COUNT:] = (1, 100, 0)
    tail, _ = filter_tracks(tail)
    assert tail[11, HOLD_START_COUNT] == 1 and tail[11, HOLD_DURATION_1] == 1
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
    print(f"[infer] 音符={int(tracks[:, :2].sum())} 后处理丢弃={dropped} 输出={path}")


if __name__ == "__main__":
    main()
