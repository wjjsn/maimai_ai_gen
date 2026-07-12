"""Sliding-window inference for maimai chart generation.

Usage:
    uv run src/infer.py <audio_file> [--checkpoint ckpts/best.pt] [--level 5] [--out output.txt]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from config import CONFIG, validate_checkpoint_config
from model import Whisper, ModelDimensions
from constrained_decode import allowed_tokens
from maidata_parser import (
    compiler, Chart, Level, Frame, Note, NoteType, TapType,
)
from tokenizer import PAD, SOS, EOS, VOCAB_SIZE, FRAME_END, decode_frames, encode_frame
from mert_cache import extract_audio_features

# ──────────────────── defaults ────────────────────

SAMPLE_RATE = 75
HOP_LENGTH = 1
N_MELS = CONFIG.model.audio_state
MAX_TOKENS = CONFIG.model.max_tokens
WINDOW_FRAMES = round(CONFIG.window.mel_frames * CONFIG.audio.hop_length / CONFIG.audio.sample_rate * SAMPLE_RATE)
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
    if ckpt.get("model_kind") != "mert-v1-95m-frozen-decoder":
        raise ValueError("检查点不是冻结 MERT-v1-95M decoder，旧 Whisper 检查点不能用于本推理路径")
    dims: ModelDimensions = ckpt["dims"]
    validate_checkpoint_config(
        ckpt.get("config"),
        CONFIG,
        for_training=False,
        allow_legacy=CONFIG.inference.allow_legacy_checkpoint,
    )
    if dims.n_audio_state != N_MELS or dims.n_text_state != N_MELS:
        raise ValueError(f"检查点状态维度与 MERT 输出不兼容: {dims.n_audio_state}/{dims.n_text_state}")
    if dims.n_audio_ctx != WINDOW_FRAMES:
        raise ValueError(f"检查点特征帧数为 {dims.n_audio_ctx}，当前配置为 {WINDOW_FRAMES}")
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


def audio_to_mel(path: str | Path, device: torch.device) -> np.ndarray:
    """提取整曲冻结 MERT 特征，返回 (T, 768)。"""
    features = extract_audio_features(path, device)
    print(f"[infer] MERT特征形状: {features.shape}")
    return features


@torch.no_grad()
def decode_segment(
    model: Whisper,
    mel_slice: torch.Tensor,   # (T, 768)
    device: torch.device,
    max_tokens: int = MAX_TOKENS,
    constrained: bool = True,
    prefix_tokens: list[int] | None = None,
    audio_mask: torch.Tensor | None = None,
    return_stats: bool = False,
    verbose: bool = True,
) -> list[int] | tuple[list[int], dict[str, int | bool]]:
    """Autoregressive greedy decode one mel segment. Returns full token list incl. SOS/EOS."""
    mel = mel_slice.unsqueeze(0).to(device)
    if audio_mask is None:
        audio_mask = torch.ones(mel_slice.shape[0], dtype=torch.bool)
    audio_mask = audio_mask.unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        audio_features = model.embed_audio(mel)

    tokens = list(prefix_tokens or [SOS])
    prefix_len = len(tokens)
    if len(tokens) >= max_tokens:
        raise ValueError(f"decoder 前缀长度 {len(tokens)} 已达到 max_tokens={max_tokens}")

    stop_reason = "token_budget"
    while len(tokens) < max_tokens - 1:
        dec_input = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, S)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.logits(dec_input, audio_features, audio_mask)   # (1, S, vocab)
        next_logits = logits[0, -1, :]   # (vocab,)
        if constrained:
            allowed = allowed_tokens(
                tokens,
                min_frame_time=round(TARGET_START_SEC * 100),
                max_frame_time=round(TARGET_END_SEC * 100) - 1,
            )
            if not allowed:
                stop_reason = "grammar_dead_end"
                break
            masked = torch.full_like(next_logits, float("-inf"))
            masked[list(allowed)] = next_logits[list(allowed)]
            next_logits = masked
        next_token = next_logits.argmax().item()

        if next_token == EOS:
            tokens.append(next_token)
            stop_reason = "eos"
            break
        if next_token == PAD:
            tokens.append(EOS)
            stop_reason = "pad"
            break
        tokens.append(next_token)

    raw_new_tokens = len(tokens) - prefix_len
    dropped_tokens = 0
    if stop_reason in ("token_budget", "grammar_dead_end") and tokens[-1] != EOS:
        # 异常终止时可能停在 TAP/Slide/frame 中间；只保留完整帧。
        try:
            last_frame_end = len(tokens) - 1 - tokens[::-1].index(FRAME_END)
        except ValueError:
            last_frame_end = len(prefix_tokens or [SOS]) - 1
        dropped_tokens = len(tokens) - last_frame_end - 1
        tokens = tokens[:last_frame_end + 1]
        tokens.append(EOS)
        if dropped_tokens and verbose:
            print(f"[infer] {stop_reason}，丢弃尾部 {dropped_tokens} 个未闭合 token")
    stats = {
        "stop_reason": stop_reason,
        "hit_limit": stop_reason == "token_budget",
        "grammar_dead_end": stop_reason == "grammar_dead_end",
        "raw_new_tokens": raw_new_tokens,
        "new_tokens": len(tokens) - prefix_len,
        "dropped_tokens": dropped_tokens,
    }
    return (tokens, stats) if return_stats else tokens


def fit_logical_window(
    mel: np.ndarray,
    logical_start_frame: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_start = max(0, logical_start_frame)
    source_end = min(mel.shape[0], logical_start_frame + window)
    left_pad = source_start - logical_start_frame
    out = np.zeros((window, N_MELS), dtype=mel.dtype)
    audio_mask = torch.zeros(window, dtype=torch.bool)
    if source_end > source_start:
        source = mel[source_start:source_end]
        out[left_pad:left_pad + source.shape[0]] = source
        audio_mask[left_pad:left_pad + source.shape[0]] = True
    return torch.from_numpy(out.copy()).float(), audio_mask


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
    total_sec = mel.shape[0] / frames_per_sec
    window_sec = window / frames_per_sec
    if abs(commit_sec - DEFAULT_COMMIT_SEC) > 1e-6:
        raise ValueError(f"当前模型窗口固定提交 {DEFAULT_COMMIT_SEC:.0f}s，不支持 commit_sec={commit_sec}")
    committed: list[Frame] = []
    limit_windows = 0
    dead_end_windows = 0
    eos_windows = 0
    total_new_tokens = 0
    total_raw_new_tokens = 0
    total_dropped_tokens = 0

    commit_start = max(0.0, start_sec)
    window_i = 0
    while commit_start < total_sec:
        window_start = commit_start - TARGET_START_SEC
        logical_start_frame = round(window_start * frames_per_sec)
        window_end_sec = window_start + window_sec
        commit_end = min(commit_start + commit_sec, total_sec)

        prefix_frames = [f for f in committed if _prefix_relative_cs(f, window_start) is not None]
        prefix_tokens = frames_to_prefix_tokens(prefix_frames, window_start)
        mel_tensor, audio_mask = fit_logical_window(mel, logical_start_frame, window)
        tokens, decode_stats = decode_segment(
            model,
            mel_tensor,
            device,
            max_tokens=max_tokens,
            prefix_tokens=prefix_tokens,
            audio_mask=audio_mask,
            return_stats=True,
            verbose=verbose,
        )
        limit_windows += int(decode_stats["hit_limit"])
        dead_end_windows += int(decode_stats["grammar_dead_end"])
        eos_windows += int(decode_stats["stop_reason"] in ("eos", "pad"))
        total_new_tokens += int(decode_stats["new_tokens"])
        total_raw_new_tokens += int(decode_stats["raw_new_tokens"])
        total_dropped_tokens += int(decode_stats["dropped_tokens"])
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
        "dead_end_windows": dead_end_windows,
        "eos_windows": eos_windows,
        "new_tokens": total_new_tokens,
        "raw_new_tokens": total_raw_new_tokens,
        "dropped_tokens": total_dropped_tokens,
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


def _self_check() -> None:
    mel = np.arange(5 * N_MELS, dtype=np.float32).reshape(5, N_MELS)
    window, mask = fit_logical_window(mel, -2, 8)
    assert window.shape == (8, N_MELS)
    assert mask.tolist() == [False, False, True, True, True, True, True, False]
    assert torch.equal(window[2:7], torch.from_numpy(mel))
    assert not window[~mask].any()

    window, mask = fit_logical_window(mel, 2, 5)
    assert mask.tolist() == [True, True, True, False, False]
    assert torch.equal(window[:3], torch.from_numpy(mel[2:]))


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

    mel = audio_to_mel(audio_path, device)
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
