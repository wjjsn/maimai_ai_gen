import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from maidata_parser import EOS, PAD, SOS, VOCAB_SIZE, compiler
from model import ModelDimensions, Whisper
from constrained_decode import allowed_tokens


SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_MELS = 80
WINDOW_FRAMES = 3000


def cache_key(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode()).hexdigest()


def pick_chart(charts_dir: Path, query: str | None) -> Path:
    paths = sorted(charts_dir.rglob("maidata.txt"))
    if query is None:
        if not paths:
            raise FileNotFoundError(f"没有找到谱面: {charts_dir}")
        return paths[0]
    matches = [p for p in paths if query in str(p.parent)]
    if not matches:
        raise FileNotFoundError(f"没有匹配谱面: {query}")
    return matches[0]


def load_one_chart(charts_dir: Path, cache_dir: Path, query: str | None, level_idx: int, max_segments: int):
    maidata_path = pick_chart(charts_dir, query)
    rel = str(maidata_path.parent.relative_to(charts_dir))
    mel_path = cache_dir / f"{cache_key(rel)}.npy"
    if not mel_path.exists():
        raise FileNotFoundError(f"缺少 mel cache: {mel_path}")

    c = compiler(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    c.parse(maidata_path.read_text(encoding="utf-8"))
    offsets, tensors = c.to_tensor(level_idx=level_idx)
    if not tensors:
        raise RuntimeError(f"谱面没有 level {level_idx}: {rel}")

    mel = np.load(mel_path)
    frames_per_sec = SAMPLE_RATE / HOP_LENGTH
    items = []
    limit = min(max_segments, len(tensors)) if max_segments > 0 else len(tensors)
    for i in range(limit):
        start = int(offsets[i] * frames_per_sec)
        if i + 1 < len(offsets):
            end = int(offsets[i + 1] * frames_per_sec)
        else:
            end = mel.shape[1]
        chunk = torch.from_numpy(mel[:, start:end].copy()).float()
        if chunk.shape[1] < WINDOW_FRAMES:
            chunk = nn.functional.pad(chunk, (0, WINDOW_FRAMES - chunk.shape[1]))
        else:
            chunk = chunk[:, :WINDOW_FRAMES]
        items.append((chunk, tensors[i]))
    return rel, offsets[:limit], items


def pad_tokens(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(r.numel()) for r in rows)
    padded = torch.full((len(rows), max_len), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        padded[i, : row.numel()] = row
        mask[i, : row.numel()] = True
    return padded, mask


@torch.no_grad()
def greedy(model: Whisper, mel: torch.Tensor, max_len: int, device: torch.device, stop: bool = True, constrained: bool = True) -> list[int]:
    model.eval()
    tokens = [SOS]
    for _ in range(max_len - 1):
        dec = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(mel.unsqueeze(0).to(device), dec)
        next_logits = logits[0, -1]
        if constrained:
            allowed = allowed_tokens(tokens)
            if not allowed:
                break
            if len(tokens) == max_len - 1:
                allowed = (EOS,) if EOS in allowed else allowed
            elif len(tokens) < max_len - 1:
                allowed = tuple(t for t in allowed if t not in (EOS, PAD)) or allowed
            mask = torch.full_like(next_logits, float("-inf"))
            mask[list(allowed)] = next_logits[list(allowed)]
            next_logits = mask
        nxt = int(next_logits.argmax())
        tokens.append(nxt)
        if stop and nxt in (EOS, PAD):
            break
    return tokens


def accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float(((pred == target) & mask).sum().item() / max(mask.sum().item(), 1))


def grammar_masked_logits(logits: torch.Tensor, dec_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    cpu_dec = dec_input.detach().cpu().tolist()
    cpu_mask = mask.detach().cpu().tolist()
    for b, row in enumerate(cpu_dec):
        prefix: list[int] = []
        for t, tok in enumerate(row):
            if not cpu_mask[b][t]:
                break
            prefix.append(tok)
            allowed = allowed_tokens(prefix)
            if allowed:
                masked[b, t, list(allowed)] = logits[b, t, list(allowed)]
    return masked


@torch.no_grad()
def greedy_score(model: Whisper, mel: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor, device: torch.device):
    rows = []
    total_same = 0
    total_len = 0
    per_segment = []
    for i in range(tokens.size(0)):
        target_len = int(mask[i].sum().item())
        gen = greedy(model, mel[i].detach().cpu(), target_len, device)
        truth = tokens[i, :target_len].detach().cpu().tolist()
        same = sum(a == b for a, b in zip(gen, truth))
        rows.append(gen)
        total_same += same
        total_len += target_len
        per_segment.append((same, target_len, len(gen)))
    return total_same / max(total_len, 1), per_segment, rows


def main():
    parser = argparse.ArgumentParser(description="单谱面过拟合验证")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--charts", type=Path, default=root / "charts")
    parser.add_argument("--cache", type=Path, default=root / ".cache" / "charts")
    parser.add_argument("--query", type=str, default="10. MURASAKi PLUS/夢花火")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--stop-on-perfect", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "best.pt")
    parser.add_argument("--save", type=Path, default=root / "checkpoints" / "overfit_one.pt")
    parser.add_argument("--out", type=Path, default=root / "tmp" / "overfit_one.txt")
    parser.add_argument("--target-out", type=Path, default=root / "tmp" / "overfit_one_target.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rel, offsets, items = load_one_chart(args.charts, args.cache, args.query, args.level, args.max_segments)
    mel = torch.stack([x for x, _ in items]).to(device)
    tokens, mask = pad_tokens([t for _, t in items])
    tokens = tokens.to(device)
    mask = mask.to(device)

    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        dims: ModelDimensions = ckpt["dims"]
        model = Whisper(dims).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[overfit] 从 checkpoint 微调: {args.checkpoint}")
    else:
        dims = ModelDimensions(
            n_mels=N_MELS,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=VOCAB_SIZE,
            n_text_ctx=max(tokens.shape[1], 8),
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )
        model = Whisper(dims).to(device)
        print("[overfit] 从随机初始化训练")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    dec_input = tokens[:, :-1]
    target = tokens[:, 1:]
    target_mask = mask[:, 1:]

    print(f"[overfit] device={device} chart={rel} segments={len(items)} token_len={tokens.shape[1]}")
    best_state = None
    best_greedy = -1.0
    best_rows = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(mel, dec_input)
        masked_logits = grammar_masked_logits(logits, dec_input, mask[:, :-1])
        loss = criterion(masked_logits.reshape(-1, masked_logits.size(-1)), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = accuracy(masked_logits, target, target_mask)
            greedy_acc, per_segment, rows = greedy_score(model, mel, tokens, mask, device)
            if greedy_acc > best_greedy:
                best_greedy = greedy_acc
                best_rows = rows
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                args.save.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"dims": dims, "model_state_dict": best_state, "chart": rel, "greedy_acc": best_greedy}, args.save)
            detail = " ".join(f"s{i}={same}/{target_len}" for i, (same, target_len, _gen_len) in enumerate(per_segment))
            print(f"[overfit] epoch={epoch:03d} loss={loss.item():.4f} tf_acc={acc * 100:.2f}% greedy_acc={greedy_acc * 100:.2f}% {detail}")
            if args.stop_on_perfect and greedy_acc >= 1.0:
                print("[overfit] greedy 已 100%，提前停止")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits = model(mel, dec_input)
        pred = torch.cat([tokens[:, :1], logits.argmax(dim=-1)], dim=1)
        final_acc = accuracy(logits, target, target_mask)

    generated_rows = best_rows if best_rows is not None else [greedy(model, mel[i].cpu(), int(mask[i].sum().item()), device) for i in range(len(items))]
    true = tokens[0].detach().cpu().tolist()
    tf_pred = pred[0].detach().cpu().tolist()
    generated = generated_rows[0]
    print(f"[overfit] final_teacher_forcing_acc={final_acc * 100:.2f}%")
    print(f"[overfit] true_first80={true[:80]}")
    print(f"[overfit] teacher_forcing_first80={tf_pred[:80]}")
    print(f"[overfit] greedy_first80={generated[:80]}")

    for i, gen in enumerate(generated_rows):
        truth = tokens[i].detach().cpu().tolist()
        truth = truth[: mask[i].sum().item()]
        n = min(len(gen), len(truth))
        same = sum(a == b for a, b in zip(gen[:n], truth[:n]))
        print(f"[overfit] segment={i} greedy_match={same}/{len(truth)} generated_len={len(gen)} target_len={len(truth)}")

    c = compiler()
    c.parse_from_tensor((offsets, [torch.tensor(row, dtype=torch.int64) for row in generated_rows]), level_idx=args.level)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(c.generate(), encoding="utf-8")
    print(f"[overfit] out={args.out}")

    target_c = compiler()
    target_c.parse_from_tensor((offsets, [row.detach().cpu() for _, row in items]), level_idx=args.level)
    args.target_out.parent.mkdir(parents=True, exist_ok=True)
    args.target_out.write_text(target_c.generate(), encoding="utf-8")
    print(f"[overfit] target_out={args.target_out}")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dims": dims, "model_state_dict": model.state_dict(), "chart": rel, "greedy_acc": best_greedy}, args.save)
    print(f"[overfit] saved={args.save}")


if __name__ == "__main__":
    main()
