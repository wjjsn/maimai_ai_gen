from config import CONFIG
from dataset import ChartDataset
from tokenizer import EOS, PAD, SOS, TOUCH_BASE, decode_frames, rotate_token_id, rotate_tokens


def strip_pad(tokens: list[int]) -> list[int]:
    while tokens and tokens[-1] == PAD:
        tokens.pop()
    return tokens


def main():
    ds = ChartDataset(
        CONFIG.paths.charts_dir,
        level_idx=CONFIG.training.level_idx,
        max_tokens=CONFIG.model.max_tokens,
    )
    tokens = strip_pad(ds[0]["tokens"].tolist())

    assert rotate_tokens(tokens, 0) == tokens
    assert rotate_tokens(tokens, 8) == tokens

    rotated = tokens
    for _ in range(8):
        rotated = rotate_tokens(rotated, 1)
    assert rotated == tokens

    touch_c = TOUCH_BASE + 16
    for steps in range(8):
        assert rotate_token_id(touch_c, steps) == touch_c

    for steps in range(8):
        row = rotate_tokens(tokens, steps)
        assert row[0] == SOS
        assert row[-1] == EOS
        decode_frames([t for t in row if t not in (SOS, EOS, PAD)])

    print("[rotation] 旋转增强自检通过")


if __name__ == "__main__":
    main()
