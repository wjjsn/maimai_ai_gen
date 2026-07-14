"""四头事件 CNN 正式推理入口。"""

import shutil
from pathlib import Path

import numpy as np
import torch

from audio_features import extract_audio_features
from chart import Chart, Frame, HoldData, Level, Note, NoteType, TapType
from config import CONFIG, checkpoint_config
from maidata_parser import generate_maidata
from model import ChartCNN, ModelDimensions
from tensor_roundtrip import HOLD_DURATION_1, HOLD_START_COUNT, TAP_COUNT, TRACK_COUNT


MODEL_KIND = "log-mel-full-song-event-cnn-v4"


def _scale() -> np.ndarray:
    duration = round(CONFIG.inference.max_duration_sec * CONFIG.audio.frames_per_sec)
    return np.array((8, 2, duration, duration), dtype=np.float32)


def load_model(path: str | Path, device: torch.device) -> tuple[ChartCNN, ModelDimensions]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != 4 or checkpoint.get("model_kind") != MODEL_KIND:
        raise ValueError("检查点来自旧架构，拒绝加载")
    if checkpoint.get("config") != checkpoint_config():
        raise ValueError("检查点的音频或模型配置与当前配置不一致")
    model = ChartCNN(checkpoint["dims"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[infer] 加载 checkpoint epoch={checkpoint.get('epoch', '?')}")
    return model, checkpoint["dims"]


@torch.no_grad()
def predict_events(model: ChartCNN, features: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(device)
    mask = torch.ones(1, tensor.shape[1], dtype=torch.bool, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        events = model(tensor, mask)[0]
    return (events.float().cpu().numpy() * _scale()).round().astype(np.int32)


def events_to_frames(events: np.ndarray) -> tuple[list[Frame], dict[str, int]]:
    if events.ndim != 2 or events.shape[1] != TRACK_COUNT:
        raise ValueError(f"事件张量形状必须为 (T, {TRACK_COUNT})")
    grouped: dict[int, list[Note]] = {}
    dropped = 0
    last_tap = -CONFIG.inference.short_min_gap_frames
    for frame, values in enumerate(events):
        tap_count = max(0, int(values[TAP_COUNT]))
        if tap_count and frame - last_tap >= CONFIG.inference.short_min_gap_frames:
            for index in range(tap_count):
                grouped.setdefault(frame, []).append(Note(NoteType.TAP, (TapType.LANE1, TapType.LANE8)[index % 2]))
            last_tap = frame
        else:
            dropped += tap_count
        for slot in range(min(2, max(0, int(values[HOLD_START_COUNT])))):
            duration_frames = int(values[HOLD_DURATION_1 + slot])
            if duration_frames <= 0:
                continue
            duration = min(CONFIG.inference.max_duration_sec, max(CONFIG.inference.min_duration_sec, duration_frames / CONFIG.audio.frames_per_sec))
            grouped.setdefault(frame, []).append(Note(NoteType.HOLD, HoldData((TapType.LANE2, TapType.LANE3)[slot], duration)))
    frames = [Frame(tuple(notes), frame / CONFIG.audio.frames_per_sec) for frame, notes in sorted(grouped.items())]
    return frames, {"events": sum(len(notes) for notes in grouped.values()), "dropped": dropped}


def infer_features(model: ChartCNN, features: np.ndarray, device: torch.device):
    events = predict_events(model, features, device)
    frames, stats = events_to_frames(events)
    return frames, {**stats, "tap_events": int(events[:, TAP_COUNT].sum()), "hold_events": int(events[:, HOLD_START_COUNT].sum())}


def frames_to_maidata(frames: list[Frame], level_idx: int, title: str = "generated") -> str:
    chart = Chart(title=title, artist="generated")
    chart.all_levels[level_idx] = Level(f"level_{level_idx + 1}", 0.0, frames)
    return generate_maidata(chart)


def save_inference_files(audio_path: Path, text: str, out_dir: Path) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"推理输出目录必须为空: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, out_dir / "track.mp3")
    generated = out_dir / "maidata.txt"
    generated.write_text(text, encoding="utf-8")
    return generated


def _self_check() -> None:
    events = np.zeros((20, TRACK_COUNT), dtype=np.int32)
    events[3] = (2, 1, 20, 0)
    frames, stats = events_to_frames(events)
    assert len(frames) == 1 and len(frames[0].notes) == 3
    assert frames[0].time_sec == 3 / CONFIG.audio.frames_per_sec
    assert stats == {"events": 3, "dropped": 0}
    print("[infer] 自检通过")


def main() -> None:
    audio_path = CONFIG.inference.audio_path
    output_dir = CONFIG.paths.inference_output_dir / audio_path.stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(CONFIG.inference.checkpoint, device)
    frames, stats = infer_features(model, extract_audio_features(audio_path, device), device)
    path = save_inference_files(audio_path, frames_to_maidata(frames, CONFIG.inference.level_idx, audio_path.stem), output_dir)
    print(f"[infer] 生成 {len(frames)} 帧，丢弃 {stats['dropped']}/{stats['events']} 个事件")
    print(f"[infer] 输出: {path}")


if __name__ == "__main__":
    main()
