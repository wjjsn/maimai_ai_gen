"""Sliding-window inference for maimai chart generation.

Usage:
    uv run src/infer.py <audio_file> [--checkpoint ckpts/best.pt] [--level 5] [--out output.txt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from model import Whisper, ModelDimensions
from constrained_decode import allowed_tokens
from maidata_parser import (
    PAD, SOS, EOS, VOCAB_SIZE, FRAME_START, FRAME_END,
    compiler, Chart, Level, Frame, Note, NoteType, TapType,
)

# ──────────────────── defaults ────────────────────

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
MAX_TOKENS = 2048
ENCODER_CTX = 1500      # encoder 期望的 mel 帧数（conv2 输出）
WINDOW_FRAMES = 3000    # 每窗 mel 帧数（conv2 stride=2 后 → ENCODER_CTX）
DEFAULT_LEVEL = 5
PREFIX_START_SEC = 6.0
TARGET_START_SEC = 12.0
TARGET_END_SEC = 22.0
DEFAULT_COMMIT_SEC = TARGET_END_SEC - TARGET_START_SEC


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[Whisper, ModelDimensions]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dims: ModelDimensions = ckpt["dims"]
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
            allowed = allowed_tokens(tokens, min_frame_time=1200, max_frame_time=2199)
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


def frames_to_prefix_tokens(frames: list[Frame], window_start_sec: float) -> list[int]:
    c = compiler()
    tokens = [SOS]
    for frame in frames:
        rel_time = frame.time_sec - window_start_sec
        if not (PREFIX_START_SEC <= rel_time < TARGET_START_SEC):
            continue
        tokens.extend((FRAME_START, c._ts_token(rel_time)))
        for note in frame.notes:
            tokens.extend(c._encode_note_tokens(note))
        tokens.append(FRAME_END)
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
    parser = compiler()
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

        prefix_start = commit_start - (TARGET_START_SEC - PREFIX_START_SEC)
        prefix_frames = [f for f in committed if prefix_start <= f.time_sec < commit_start]
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
        rel_frames = parser._parse_token_segment(generated_body)

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    parser.add_argument("--out", type=str, default=None, help="Output text file path")
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--commit-sec", type=float, default=DEFAULT_COMMIT_SEC)
    parser.add_argument("--start-sec", type=float, default=0.0)
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
    frames = overlap_infer(model, mel, device, window=args.window, commit_sec=args.commit_sec, start_sec=args.start_sec)
    print(f"[infer] total frames: {len(frames)}")
    maidata_text = frames_to_maidata(frames, level_idx=args.level)

    if args.out:
        out_path = Path(args.out)
    else:
        tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
        out_path = tmp_dir.with_suffix(".txt")
    out_path.write_text(maidata_text, encoding="utf-8")
    print(f"[infer] saved to '{out_path}'")


if __name__ == "__main__":
    main()
