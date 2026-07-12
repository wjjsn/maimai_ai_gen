"""Sliding-window inference for maimai chart generation.

Usage:
    uv run src/infer.py <audio_file> [--checkpoint ckpts/best.pt] [--level 5] [--out output.txt]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio

from config import CONFIG, validate_checkpoint_config
from model import Whisper, ModelDimensions
from constrained_decode import allowed_tokens
from maidata_parser import (
    compiler, Chart, Level, Frame, Note, NoteType, TapType,
)
from tokenizer import PAD, SOS, EOS, VOCAB_SIZE, FRAME_END, decode_frames, encode_frame

# ──────────────────── defaults ────────────────────

SAMPLE_RATE = CONFIG.audio.sample_rate
N_FFT = CONFIG.audio.n_fft
HOP_LENGTH = CONFIG.audio.hop_length
N_MELS = CONFIG.audio.n_mels
MAX_TOKENS = CONFIG.model.max_tokens
WINDOW_FRAMES = CONFIG.window.mel_frames
DEFAULT_LEVEL = CONFIG.inference.level_idx
PREFIX_START_SEC = CONFIG.window.prefix_start_sec
TARGET_START_SEC = CONFIG.window.target_start_sec
TARGET_END_SEC = CONFIG.window.target_end_sec
DEFAULT_COMMIT_SEC = CONFIG.window.infer_stride_sec


def _level_arg(value: str) -> int:
    level = int(value)
    if not 0 <= level <= 6:
        raise argparse.ArgumentTypeError("难度编号必须在 0 到 6 之间")
    return level


def _nonnegative_float_arg(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的有限数字")
    return number


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[Whisper, ModelDimensions]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dims: ModelDimensions = ckpt["dims"]
    validate_checkpoint_config(
        ckpt.get("config"),
        CONFIG,
        for_training=False,
        allow_legacy=CONFIG.inference.allow_legacy_checkpoint,
    )
    if dims.n_mels != N_MELS:
        raise ValueError(f"检查点梅尔频带数为 {dims.n_mels}，当前配置为 {N_MELS}")
    if dims.n_audio_ctx * 2 != WINDOW_FRAMES:
        raise ValueError(f"检查点梅尔帧数为 {dims.n_audio_ctx * 2}，当前配置为 {WINDOW_FRAMES}")
    if dims.n_text_ctx != MAX_TOKENS:
        raise ValueError(f"检查点最大词元数为 {dims.n_text_ctx}，当前配置为 {MAX_TOKENS}")
    model = Whisper(dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    expected = {
        "mel_frames": WINDOW_FRAMES,
        "prefix_start_sec": PREFIX_START_SEC,
        "target_start_sec": TARGET_START_SEC,
        "target_end_sec": TARGET_END_SEC,
        "infer_stride_sec": DEFAULT_COMMIT_SEC,
    }
    config = ckpt.get("window_config")
    if config is not None:
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"checkpoint 窗口配置不兼容: {key}={config.get(key)!r}, 当前需要 {value!r}")
    print(f"[infer] loaded checkpoint epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")
    return model, dims


def audio_to_mel(path: str | Path) -> np.ndarray:
    """Load audio → log-mel spectrogram, returns (n_mels, T)."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel = transform(waveform)
    log_mel = torch.log(mel + 1e-6).squeeze(0).numpy()
    print(f"[infer] mel shape: {log_mel.shape} (n_mels={N_MELS}, frames={log_mel.shape[1]})")
    return log_mel


@torch.no_grad()
def decode_segment(
    model: Whisper,
    mel_slice: torch.Tensor,   # (n_mels, T) — 已经 fit 到 encoder_ctx 帧
    device: torch.device,
    max_tokens: int = MAX_TOKENS,
    constrained: bool = True,
    prefix_tokens: list[int] | None = None,
    return_stats: bool = False,
    verbose: bool = True,
) -> list[int] | tuple[list[int], dict[str, int | bool]]:
    """Autoregressive greedy decode one mel segment. Returns full token list incl. SOS/EOS."""
    mel = mel_slice.unsqueeze(0).to(device)   # (1, n_mels, T)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        audio_features = model.embed_audio(mel)

    tokens = list(prefix_tokens or [SOS])
    prefix_len = len(tokens)
    if len(tokens) >= max_tokens:
        raise ValueError(f"decoder 前缀长度 {len(tokens)} 已达到 max_tokens={max_tokens}")

    hit_limit = True
    while len(tokens) < max_tokens - 1:
        dec_input = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, S)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.logits(dec_input, audio_features)   # (1, S, vocab)
        next_logits = logits[0, -1, :]   # (vocab,)
        if constrained:
            allowed = allowed_tokens(
                tokens,
                min_frame_time=round(TARGET_START_SEC * 100),
                max_frame_time=round(TARGET_END_SEC * 100) - 1,
            )
            if not allowed:
                break
            masked = torch.full_like(next_logits, float("-inf"))
            masked[list(allowed)] = next_logits[list(allowed)]
            next_logits = masked
        next_token = next_logits.argmax().item()

        if next_token == EOS:
            tokens.append(next_token)
            hit_limit = False
            break
        if next_token == PAD:
            tokens.append(EOS)
            hit_limit = False
            break
        tokens.append(next_token)

    if hit_limit and tokens[-1] != EOS:
        # token 预算耗尽时可能停在 TAP/Slide/frame 中间；只保留完整帧。
        try:
            last_frame_end = len(tokens) - 1 - tokens[::-1].index(FRAME_END)
        except ValueError:
            last_frame_end = len(prefix_tokens or [SOS]) - 1
        dropped = len(tokens) - last_frame_end - 1
        tokens = tokens[:last_frame_end + 1]
        tokens.append(EOS)
        if dropped and verbose:
            print(f"[infer] token 达到上限，丢弃尾部 {dropped} 个未闭合 token")
    stats = {
        "hit_limit": hit_limit,
        "new_tokens": len(tokens) - prefix_len,
    }
    return (tokens, stats) if return_stats else tokens


def fit_logical_window(mel: np.ndarray, logical_start_frame: int, window: int) -> torch.Tensor:
    source_start = max(0, logical_start_frame)
    source_end = min(mel.shape[1], logical_start_frame + window)
    left_pad = source_start - logical_start_frame
    out = np.zeros((N_MELS, window), dtype=mel.dtype)
    if source_end > source_start:
        source = mel[:, source_start:source_end]
        out[:, left_pad:left_pad + source.shape[1]] = source
    return torch.from_numpy(out.copy()).float()


def _prefix_relative_cs(frame: Frame, window_start_sec: float) -> int | None:
    """与训练缓存一致：先量化到厘秒，再判断前缀区间。"""
    rel_cs = round((frame.time_sec - window_start_sec) * 100)
    return rel_cs if round(PREFIX_START_SEC * 100) <= rel_cs < round(TARGET_START_SEC * 100) else None


def frames_to_prefix_tokens(frames: list[Frame], window_start_sec: float) -> list[int]:
    tokens = [SOS]
    for frame in frames:
        rel_cs = _prefix_relative_cs(frame, window_start_sec)
        if rel_cs is None:
            continue
        tokens.extend(encode_frame(frame, rel_cs / 100.0))
    return tokens


def overlap_infer(
    model: Whisper,
    mel: np.ndarray,
    device: torch.device,
    window: int = WINDOW_FRAMES,
    commit_sec: float = DEFAULT_COMMIT_SEC,
    start_sec: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    verbose: bool = True,
    return_stats: bool = False,
) -> list[Frame] | tuple[list[Frame], dict[str, int]]:
    """使用上一窗口最后 6 秒 token 作为前缀，生成中间 10 秒。"""
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    total_sec = mel.shape[1] / frames_per_sec
    window_sec = window / frames_per_sec
    if abs(commit_sec - DEFAULT_COMMIT_SEC) > 1e-6:
        raise ValueError(f"当前模型窗口固定提交 {DEFAULT_COMMIT_SEC:.0f}s，不支持 commit_sec={commit_sec}")
    committed: list[Frame] = []
    limit_windows = 0
    total_new_tokens = 0

    commit_start = max(0.0, start_sec)
    window_i = 0
    while commit_start < total_sec:
        window_start = commit_start - TARGET_START_SEC
        logical_start_frame = round(window_start * frames_per_sec)
        window_end_sec = window_start + window_sec
        commit_end = min(commit_start + commit_sec, total_sec)

        prefix_frames = [f for f in committed if _prefix_relative_cs(f, window_start) is not None]
        prefix_tokens = frames_to_prefix_tokens(prefix_frames, window_start)
        mel_tensor = fit_logical_window(mel, logical_start_frame, window)
        tokens, decode_stats = decode_segment(
            model,
            mel_tensor,
            device,
            max_tokens=max_tokens,
            prefix_tokens=prefix_tokens,
            return_stats=True,
            verbose=verbose,
        )
        limit_windows += int(decode_stats["hit_limit"])
        total_new_tokens += int(decode_stats["new_tokens"])
        generated_body = [t for t in tokens[len(prefix_tokens):] if t not in (SOS, EOS, PAD)]
        rel_frames = decode_frames(generated_body)

        accepted = 0
        for frame in rel_frames:
            if not (TARGET_START_SEC <= frame.time_sec < TARGET_END_SEC):
                continue
            abs_time = window_start + frame.time_sec
            if not (commit_start <= abs_time < commit_end):
                continue
            f = Frame(notes=frame.notes, time_sec=abs_time)
            committed.append(f)
            accepted += 1

        if verbose:
            print(
                f"[infer]   overlap window {window_i} {window_start:.1f}..{window_end_sec:.1f}s "
                f"prefix={len(prefix_frames)} commit {commit_start:.1f}..{commit_end:.1f}s "
                f"frames={len(rel_frames)} kept={accepted}"
            )

        commit_start += commit_sec
        window_i += 1

    frames = sorted(committed, key=lambda f: f.time_sec)
    stats = {
        "windows": window_i,
        "limit_windows": limit_windows,
        "new_tokens": total_new_tokens,
    }
    return (frames, stats) if return_stats else frames


def frames_to_maidata(frames: list[Frame], level_idx: int) -> str:
    c = compiler()
    c.chart = Chart(all_levels=[None] * 7, title="generated", artist="default")
    c.chart.all_levels[level_idx] = Level(
        level_name=f"level_{level_idx + 1}",
        level_query=0.0,
        frames=sorted(frames, key=lambda f: f.time_sec),
    )
    return c.generate()


def main():
    parser = argparse.ArgumentParser(description="maimai chart inference")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--checkpoint", type=Path, default=CONFIG.inference.checkpoint)
    parser.add_argument("--level", type=_level_arg, default=DEFAULT_LEVEL)
    parser.add_argument("--out", type=str, default=None, help="Output text file path")
    parser.add_argument("--start-sec", type=_nonnegative_float_arg, default=CONFIG.inference.start_sec)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[infer] device: {device}")

    model, dims = load_model(args.checkpoint, device)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[infer] ERROR: {audio_path} not found")
        return

    mel = audio_to_mel(audio_path)
    frames = overlap_infer(model, mel, device, start_sec=args.start_sec)
    print(f"[infer] total frames: {len(frames)}")
    maidata_text = frames_to_maidata(frames, level_idx=args.level)

    if args.out:
        out_path = Path(args.out)
    else:
        tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
        out_path = tmp_dir.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(maidata_text, encoding="utf-8")
    print(f"[infer] saved to '{out_path}'")


if __name__ == "__main__":
    main()
