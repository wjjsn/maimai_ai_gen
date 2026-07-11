from pathlib import Path

from dataset import ChartDataset, rotate_token_id, rotate_token_list
from maidata_parser import EOS, PAD, SOS, TOUCH_BASE, compiler


def strip_pad(tokens: list[int]) -> list[int]:
    while tokens and tokens[-1] == PAD:
        tokens.pop()
    return tokens


def main():
    root = Path(__file__).resolve().parent.parent
    ds = ChartDataset(root / "charts", level_idx=5, max_tokens=2048)
    tokens = strip_pad(ds[0]["tokens"].tolist())

    assert rotate_token_list(tokens, 0) == tokens
    assert rotate_token_list(tokens, 8) == tokens

    rotated = tokens
    for _ in range(8):
        rotated = rotate_token_list(rotated, 1)
    assert rotated == tokens

    touch_c = TOUCH_BASE + 16
    for steps in range(8):
        assert rotate_token_id(touch_c, steps) == touch_c

    c = compiler()
    for steps in range(8):
        row = rotate_token_list(tokens, steps)
        assert row[0] == SOS
        assert row[-1] == EOS
        c._parse_token_segment([t for t in row if t not in (SOS, EOS, PAD)])

    print("[rotation] 旋转增强自检通过")


if __name__ == "__main__":
    main()
