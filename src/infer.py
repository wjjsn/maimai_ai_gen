"""Sliding-window inference for maimai chart generation.

Usage:
    uv run src/infer.py <audio_file> [--checkpoint ckpts/best.pt] [--level 4] [--out output.txt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from model import Whisper, ModelDimensions
from constrained_decode import allowed_tokens
from maidata_parser import (
    PAD, SOS, EOS, VOCAB_SIZE,
    compiler, Chart, Level, Frame, Note, NoteType, TapType,
)

# ──────────────────── defaults ────────────────────

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
MAX_TOKENS = 2048
ENCODER_CTX = 1500      # encoder 期望的 mel 帧数（conv2 输出）
SLIDE_HOP = 2400        # 滑窗步进帧数（≈27.8s，留 600 帧重叠）
WINDOW_FRAMES = 3000    # 每窗 mel 帧数（conv2 stride=2 后 → ENCODER_CTX）
DEFAULT_LEVEL = 4
DEFAULT_CONTEXT_SEC = 5.0


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[Whisper, ModelDimensions]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dims: ModelDimensions = ckpt["dims"]
    model = Whisper(dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
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
) -> list[int]:
    """Autoregressive greedy decode one mel segment. Returns full token list incl. SOS/EOS."""
    mel = mel_slice.unsqueeze(0).to(device)   # (1, n_mels, T)

    tokens: list[int] = [SOS]

    for _ in range(max_tokens):
        dec_input = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, S)
        logits = model(mel, dec_input)   # (1, S, vocab)
        next_logits = logits[0, -1, :]   # (vocab,)
        if constrained:
            allowed = allowed_tokens(tokens)
            if not allowed:
                break
            masked = torch.full_like(next_logits, float("-inf"))
            masked[list(allowed)] = next_logits[list(allowed)]
            next_logits = masked
        next_token = next_logits.argmax().item()

        if next_token == EOS:
            tokens.append(next_token)
            break
        if next_token == PAD:
            tokens.append(EOS)
            break
        tokens.append(next_token)

    if tokens[-1] != EOS:
        tokens.append(EOS)
    return tokens


def fit_window(chunk: np.ndarray, window: int) -> torch.Tensor:
    n_frames = chunk.shape[1]
    if n_frames < window:
        pad = np.zeros((N_MELS, window - n_frames), dtype=chunk.dtype)
        chunk = np.concatenate([chunk, pad], axis=1)
    else:
        chunk = chunk[:, :window]
    return torch.from_numpy(chunk.copy()).float()


def slide_infer(
    model: Whisper,
    mel: np.ndarray,       # (n_mels, T_total)
    device: torch.device,
    window: int = WINDOW_FRAMES,
    hop: int = SLIDE_HOP,
    max_tokens: int = MAX_TOKENS,
) -> list[tuple[float, list[int]]]:
    """Slide a window over the full mel, decode each chunk.

    Returns list of (segment_offset_sec, tokens) per §6.3.
    Each segment's tokens have relative timestamps (TS relative to segment start).
    """
    T = mel.shape[1]
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    segments: list[tuple[float, list[int]]] = []
    pos = 0

    while pos < T:
        end = min(pos + window, T)
        chunk = mel[:, pos:end]

        offset_sec = pos / frames_per_sec
        mel_tensor = fit_window(chunk, window)
        seg_tokens = decode_segment(model, mel_tensor, device, max_tokens)
        segments.append((offset_sec, seg_tokens))

        print(f"[infer]   window {pos}..{end} (t={offset_sec:.1f}s)  →  {len(seg_tokens)} tokens")

        if end >= T:
            break
        pos += hop

    return segments


def reference_infer(
    model: Whisper,
    mel: np.ndarray,
    device: torch.device,
    reference_chart: Path,
    level_idx: int,
    window: int = WINDOW_FRAMES,
) -> list[tuple[float, list[int]]]:
    c = compiler(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    c.parse(reference_chart.read_text(encoding="utf-8"))
    offsets, target_tensors = c.to_tensor(level_idx=level_idx)
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    segments: list[tuple[float, list[int]]] = []

    for i, target in enumerate(target_tensors):
        start = int(offsets[i] * frames_per_sec)
        if i + 1 < len(offsets):
            end = int(offsets[i + 1] * frames_per_sec)
        else:
            end = mel.shape[1]
        mel_tensor = fit_window(mel[:, start:end], window)
        seg_tokens = decode_segment(model, mel_tensor, device, max_tokens=int(target.numel()))
        segments.append((offsets[i], seg_tokens))
        same = sum(a == b for a, b in zip(seg_tokens, target.tolist()))
        print(
            f"[infer]   ref segment {i} offset={offsets[i]:.3f}s "
            f"match={same}/{target.numel()} tokens={len(seg_tokens)}"
        )

    return segments


def _frame_key(frame: Frame) -> tuple:
    notes = []
    for note in frame.notes:
        notes.append(repr(note))
    return (round(frame.time_sec / 0.02), tuple(notes))


def overlap_infer(
    model: Whisper,
    mel: np.ndarray,
    device: torch.device,
    window: int = WINDOW_FRAMES,
    context_sec: float = DEFAULT_CONTEXT_SEC,
    start_sec: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> list[Frame]:
    """重叠上下文推理：窗口重叠，只提交中心区域的 frame。"""
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    total_sec = mel.shape[1] / frames_per_sec
    window_sec = window / frames_per_sec
    stride_sec = max(1.0, window_sec - 2 * context_sec)
    stride_frames = max(1, int(stride_sec * frames_per_sec))
    parser = compiler()
    kept: dict[tuple, tuple[float, Frame]] = {}

    pos = max(0, int(start_sec * frames_per_sec))
    window_i = 0
    while pos < mel.shape[1]:
        end = min(pos + window, mel.shape[1])
        offset_sec = pos / frames_per_sec
        window_end_sec = min(offset_sec + window_sec, total_sec)
        commit_start = offset_sec if window_i == 0 else offset_sec + context_sec
        commit_end = total_sec if end >= mel.shape[1] else window_end_sec - context_sec
        center = (commit_start + commit_end) / 2.0

        mel_tensor = fit_window(mel[:, pos:end], window)
        tokens = decode_segment(model, mel_tensor, device, max_tokens=max_tokens)
        rel_frames = parser._parse_token_segment(tokens)

        accepted = 0
        for frame in rel_frames:
            abs_time = offset_sec + frame.time_sec
            if not (commit_start <= abs_time < commit_end):
                continue
            f = Frame(notes=frame.notes, time_sec=abs_time)
            key = _frame_key(f)
            score = -abs(abs_time - center)
            old = kept.get(key)
            if old is None or score > old[0]:
                kept[key] = (score, f)
            accepted += 1

        print(
            f"[infer]   overlap window {window_i} {offset_sec:.1f}..{window_end_sec:.1f}s "
            f"commit {commit_start:.1f}..{commit_end:.1f}s frames={len(rel_frames)} kept={accepted}"
        )

        if end >= mel.shape[1]:
            break
        pos += stride_frames
        window_i += 1

    return [f for _score, f in sorted(kept.values(), key=lambda x: x[1].time_sec)]


def segments_to_maidata(segments: list[tuple[float, list[int]]], level_idx: int) -> str:
    """Convert segmented tokens to maidata text using the compiler.

    Each segment carries its absolute_time_offset per §6.1/§6.5.
    """
    c = compiler()
    offsets = [off for off, _ in segments]
    tensors = [torch.tensor(toks, dtype=torch.int64) for _, toks in segments]
    c.parse_from_tensor((offsets, tensors), level_idx=level_idx)
    if c.chart is None or not c.chart.all_levels or c.chart.all_levels[level_idx] is None:
        return "# [warning] no frames decoded\n"
    return c.generate()


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
    parser.add_argument("--hop", type=int, default=SLIDE_HOP)
    parser.add_argument("--reference-chart", type=str, default=None, help="Use reference chart offsets/lengths for overfit validation")
    parser.add_argument("--mode", choices=("overlap", "fixed"), default="overlap")
    parser.add_argument("--context-sec", type=float, default=DEFAULT_CONTEXT_SEC)
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
    if args.reference_chart:
        segments = reference_infer(model, mel, device, Path(args.reference_chart), args.level, window=args.window)
        total_tokens = sum(len(toks) for _, toks in segments)
        print(f"[infer] total segments: {len(segments)}, tokens: {total_tokens}")
        maidata_text = segments_to_maidata(segments, level_idx=args.level)
    elif args.mode == "overlap":
        frames = overlap_infer(model, mel, device, window=args.window, context_sec=args.context_sec, start_sec=args.start_sec)
        print(f"[infer] total frames: {len(frames)}")
        maidata_text = frames_to_maidata(frames, level_idx=args.level)
    else:
        segments = slide_infer(model, mel, device, window=args.window, hop=args.hop)
        total_tokens = sum(len(toks) for _, toks in segments)
        print(f"[infer] total segments: {len(segments)}, tokens: {total_tokens}")
        maidata_text = segments_to_maidata(segments, level_idx=args.level)

    if args.out:
        out_path = Path(args.out)
    else:
        tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
        out_path = tmp_dir.with_suffix(".txt")
    out_path.write_text(maidata_text, encoding="utf-8")
    print(f"[infer] saved to '{out_path}'")


if __name__ == "__main__":
    main()
