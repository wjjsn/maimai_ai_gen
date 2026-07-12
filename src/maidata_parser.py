import json
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from urllib.request import urlopen

import torch

from tokenizer import (
    EOS, FRAME_END, FRAME_START, IS_BREAK, IS_CCW, IS_CW, IS_EX, IS_FAKE_ROTATE,
    IS_FIREWORK, IS_FORCE_STAR, IS_SLIDE_BREAK, IS_SLIDE_NO_HEAD, LANE_BASE,
    NOTE_HOLD, NOTE_SLIDE, NOTE_TAP, NOTE_TOUCH, PAD, SEGMENT_END, SEGMENT_START,
    SLIDE_SHAPE_BASE, SOS, TOUCH_BASE, TS_BASE, VOCAB_SIZE,
)


# ── Note ──────────────────────────────────────────────────────────────────


class NoteType(Enum):
    TAP = auto()
    HOLD = auto()
    SLIDE = auto()
    TOUCH = auto()
    TOUCH_HOLD = auto()


class TouchType(Enum):
    A1 = auto()
    A2 = auto()
    A3 = auto()
    A4 = auto()
    A5 = auto()
    A6 = auto()
    A7 = auto()
    A8 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    B4 = auto()
    B5 = auto()
    B6 = auto()
    B7 = auto()
    B8 = auto()
    C = auto()
    D1 = auto()
    D2 = auto()
    D3 = auto()
    D4 = auto()
    D5 = auto()
    D6 = auto()
    D7 = auto()
    D8 = auto()
    E1 = auto()
    E2 = auto()
    E3 = auto()
    E4 = auto()
    E5 = auto()
    E6 = auto()
    E7 = auto()
    E8 = auto()


class TapType(Enum):
    LANE1 = auto()
    LANE2 = auto()
    LANE3 = auto()
    LANE4 = auto()
    LANE5 = auto()
    LANE6 = auto()
    LANE7 = auto()
    LANE8 = auto()


class SlideShape(Enum):
    Line = auto()  # -
    Circle = auto()  # > < ^
    V = auto()  # v
    GrandV = auto()  # V (L型折线)
    P = auto()  # p (单弯)
    Q = auto()  # q (单弯镜像)
    PP = auto()  # pp (大弯)
    QQ = auto()  # qq (大弯镜像)
    S = auto()  # s (闪电)
    Z = auto()  # z (闪电镜像)
    Wifi = auto()  # w (扇形/WiFi)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class SlideSegment:
    shape: SlideShape
    start_lane: TapType
    end_lane: TapType
    wait_duration: float = 0.0  # 等待时间（秒），默认一拍
    trace_duration: float = 0.0  # 持续时间（秒）

    isClockwise: bool | None = None  # > 顺时针 / < 逆时针 / ^ 自动判断选最短，不需要这个字段
    middle_lane: TapType | None = None  # GrandV的拐点。1V35  →  起点=1, 拐点=3, 终点=5

    isForceStar: bool = False  # 强制星星头
    isFakeRotate: bool = False  # 假旋转 ($$)
    isSlideBreak: bool = False  # Slide 本体 Break。b 在 ] 后 = 整条 Slide 变 Break
    isSlideNoHead: bool = False  # Slide 无头 (! 或 ? 或 *)


@dataclass
class Touch_data:
    Touch_area: TouchType
    isFirework: bool = False  # 烟花
    holdTime: float = 0.0  # 持续时间（秒），TouchHold 才有


@dataclass
class Hold_data:
    lane: TapType
    holdTime: float = 0.0  # 持续时间（秒）


@dataclass
class Note:
    """一个音符。"""
    type: NoteType
    data: TapType | Hold_data | Touch_data | list[SlideSegment]
    isBreak: bool = False
    isEx: bool = False


@dataclass
class Frame:
    """一个时间帧，包含该时间点的所有音符。"""
    notes: tuple[Note, ...] = ()
    time_sec: float = 0.0  # 精确时间（秒）


@dataclass
class Level:
    """一个难度等级，包含该难度的所有时间帧。"""
    level_name: str
    level_query: float | None
    frames: list[Frame] = field(default_factory=list)


@dataclass
class Chart:
    """一首歌的+不同难度（0.0~15.0）谱面。"""
    all_levels: list[Level | None] = field(default_factory=list)
    title: str = "default"
    artist: str = "default"
    designer: str = "default"

    # EASY/BASIC/ADVANCED/EXPERT/MASTER/Re:MASTER/ORIGINAL
    # lv_1/lv_2 /lv_3   /lv_4/  lv_5/     lv_6/  lv_7


# ── helpers ────────────────────────────────────────────────────────────────

_MUSIC_DATA_URL = "https://www.diving-fish.com/api/maimaidxprober/music_data"


def _music_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".cache" / "music_data.json"


def load_music_data() -> list[dict]:
    """只读取已下载的本地歌曲表；解析谱面绝不访问网络。"""
    try:
        data = json.loads(_music_data_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def update_music_data() -> None:
    """显式下载歌曲表，供索引筛选等离线流程使用。"""
    with urlopen(_MUSIC_DATA_URL, timeout=30) as response:
        data = json.load(response)
    if not isinstance(data, list):
        raise ValueError("歌曲表格式错误")
    path = _music_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def music_data_version() -> str:
    """本地歌曲表内容摘要；更新歌曲表才会改变训练索引。"""
    data = json.dumps(load_music_data(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _local_level_names(text: str) -> dict[int, str]:
    return {
        int(match.group(1)): match.group(2).strip()
        for match in re.finditer(r"^&lv_(\d)=([^\n]*)$", text, re.MULTILINE)
    }


def _fallback_level_name(level: str) -> str:
    try:
        value = float(level)
    except ValueError:
        return level
    if value == 13.0:
        return "13"
    if value == 13.5:
        return "13+"
    return level


def _level_to_float(level: str) -> float | None:
    try:
        return float(level)
    except ValueError:
        if level.endswith("+"):
            try:
                return float(level[:-1]) + 0.5
            except ValueError:
                pass
    return None


def _levels_match(song: dict, local_levels: dict[int, str], *, fallback: bool) -> bool:
    compared = False
    for level_idx, local_level in local_levels.items():
        ds_index = level_idx - 2
        levels = song.get("level", [])
        if not 0 <= ds_index < len(levels):
            continue
        compared = True
        expected = _fallback_level_name(local_level) if fallback else local_level
        if expected != levels[ds_index]:
            return False
    return compared


def _match_music(text: str, title: str, songs: list[dict]) -> dict | None:
    shortid_match = re.search(r"^&shortid=(\d+)\s*$", text, re.MULTILINE)
    if shortid_match:
        shortid = shortid_match.group(1)
        song = next((song for song in songs if str(song.get("id")) == shortid), None)
        if song is not None:
            return song
    matches = [song for song in songs if song.get("title") == title]
    local_levels = _local_level_names(text)
    for fallback in (False, True):
        matched = [song for song in matches if _levels_match(song, local_levels, fallback=fallback)]
        if len(matched) == 1:
            return matched[0]
    return None

_LANE_MAP = {
    "1": TapType.LANE1, "2": TapType.LANE2, "3": TapType.LANE3, "4": TapType.LANE4,
    "5": TapType.LANE5, "6": TapType.LANE6, "7": TapType.LANE7, "8": TapType.LANE8,
}

_TOUCH_MAP = {
    "A1": TouchType.A1, "A2": TouchType.A2, "A3": TouchType.A3, "A4": TouchType.A4,
    "A5": TouchType.A5, "A6": TouchType.A6, "A7": TouchType.A7, "A8": TouchType.A8,
    "B1": TouchType.B1, "B2": TouchType.B2, "B3": TouchType.B3, "B4": TouchType.B4,
    "B5": TouchType.B5, "B6": TouchType.B6, "B7": TouchType.B7, "B8": TouchType.B8,
    "C": TouchType.C, "C1": TouchType.C, "C2": TouchType.C,
    "D1": TouchType.D1, "D2": TouchType.D2, "D3": TouchType.D3, "D4": TouchType.D4,
    "D5": TouchType.D5, "D6": TouchType.D6, "D7": TouchType.D7, "D8": TouchType.D8,
    "E1": TouchType.E1, "E2": TouchType.E2, "E3": TouchType.E3, "E4": TouchType.E4,
    "E5": TouchType.E5, "E6": TouchType.E6, "E7": TouchType.E7, "E8": TouchType.E8,
}

# slide shape char(s) -> SlideShape
_SHAPE_MAP: dict[str, SlideShape] = {
    "-": SlideShape.Line,
    ">": SlideShape.Circle, "<": SlideShape.Circle, "^": SlideShape.Circle,
    "v": SlideShape.V,
    "V": SlideShape.GrandV,
    "p": SlideShape.P,
    "q": SlideShape.Q,
    "pp": SlideShape.PP,
    "qq": SlideShape.QQ,
    "s": SlideShape.S,
    "z": SlideShape.Z,
    "w": SlideShape.Wifi,
}

# ordered longest-first so "pp" matches before "p"
_SHAPE_TOKENS = sorted(_SHAPE_MAP.keys(), key=len, reverse=True)


def _lane(ch: str) -> TapType:
    return _LANE_MAP[ch]


def _touch_area(s: str) -> TouchType:
    return _TOUCH_MAP.get(s, TouchType.C)


def _length_to_seconds(length_str: str, bpm: float) -> float:
    """
    Parse a simai length specifier like '4:3', '#5.678', '150#2:1', '3##1.5', '3##8:3', '3##160#8:3'
    Returns duration in seconds.
    """
    if not length_str:
        return 0.0

    # split off optional leading wait_seconds##
    wait_seconds = 0.0
    rest = length_str
    if "##" in length_str:
        parts = length_str.split("##", 1)
        wait_seconds = float(parts[0]) if parts[0] else 0.0
        rest = parts[1]

    # rest can be:  '#SECONDS'  or  'BPM#divider:mult'  or  'divider:mult'
    if rest.startswith("#"):
        # absolute seconds: #5.678
        trace_seconds = float(rest[1:])
        return wait_seconds + trace_seconds

    # check for BPM#...
    trace_bpm = bpm
    if "#" in rest:
        bp, rest = rest.split("#", 1)
        trace_bpm = float(bp)

    # rest is now 'divider:mult'
    if ":" in rest:
        divider_s, mult_s = rest.split(":", 1)
        divider = float(divider_s)
        mult = float(mult_s)
        if trace_bpm > 0 and divider > 0:
            return wait_seconds + (240.0 / trace_bpm / divider) * mult
    else:
        # plain number treated as divider with mult=1
        divider = float(rest)
        if trace_bpm > 0 and divider > 0:
            return wait_seconds + 240.0 / trace_bpm / divider

    return wait_seconds


def _touch_hold_length(token: str, bpm: float) -> float | None:
    """Extract hold length (seconds) from a touch-hold token. Returns None for pseudo-hold."""
    m = re.search(r"\[([^\]]+)\]", token)
    if not m:
        return None  # pseudo-hold
    return _length_to_seconds(m.group(1), bpm)


# ── compiler ──────────────────────────────────────────────────────────────

class compiler:
    def __init__(self, hop_length=512, sample_rate=44100):
        self.chart = Chart(all_levels=[None] * 7)
        self.current_time = 0.0  # seconds

    # ── note parsing ──────────────────────────────────────────────────────

    def _parse_single_note(self, token: str, bpm: float) -> Note | None:
        """Parse a single (non-EACH) note token into a Note."""
        if not token or token == "E":
            return None

        # ── TOUCH / TOUCH_HOLD (A-E prefix) ──
        if token[0] in "ABCDE":
            return self._parse_touch(token, bpm)

        # ── SLIDE (contains shape chars) ──
        if self._is_slide(token):
            return self._parse_slide(token, bpm)

        # ── HOLD (contains 'h') ──
        if "h" in token:
            return self._parse_hold(token, bpm)

        # ── TAP ──
        return self._parse_tap(token)

    @staticmethod
    def _is_multi_tap(token: str) -> bool:
        """Check if token is a compact EACH of pure TAPs (e.g. '17' = TAP1 + TAP7)."""
        return len(token) >= 2 and all(c in _LANE_MAP for c in token)

    def _parse_note(self, token: str, bpm: float) -> list[Note] | None:
        """Parse note token, handling EACH (``/``). Returns list of Notes."""
        if not token or token == "E":
            return None
        parts = token.split("/")
        notes: list[Note] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # compact EACH: '17' -> TAP at 1 + TAP at 7
            if self._is_multi_tap(part):
                for ch in part:
                    notes.append(Note(type=NoteType.TAP, data=_lane(ch)))
            else:
                n = self._parse_single_note(part, bpm)
                if n is not None:
                    notes.append(n)
        return notes if notes else None

    # ── individual note type parsers ──────────────────────────────────────

    def _parse_tap(self, token: str) -> Note:
        if token[0] not in _LANE_MAP:
            return Note(type=NoteType.TAP, data=TapType.LANE1)  # fallback
        lane = _lane(token[0])
        is_break = "b" in token
        is_ex = "x" in token
        return Note(type=NoteType.TAP, data=lane, isBreak=is_break, isEx=is_ex)

    def _parse_hold(self, token: str, bpm: float) -> Note:
        lane = _lane(token[0])
        is_break = "b" in token
        is_ex = "x" in token
        m = re.search(r"\[([^\]]+)\]", token)
        if m:
            hold_sec = _length_to_seconds(m.group(1), bpm)
            return Note(
                type=NoteType.HOLD,
                data=Hold_data(lane=lane, holdTime=hold_sec),
                isBreak=is_break,
                isEx=is_ex,
            )
        else:
            # pseudo-hold (no bracket): treat as TAP
            return Note(
                type=NoteType.TAP,
                data=lane,
                isBreak=is_break,
                isEx=is_ex,
            )

    def _parse_touch(self, token: str, bpm: float) -> Note:
        has_h = "h" in token
        has_f = "f" in token

        # extract area: first char + optional digit
        if len(token) > 1 and token[1].isdigit():
            area_str = token[:2]
        else:
            area_str = token[:1]
        area = _touch_area(area_str)

        if has_h:
            m = re.search(r"\[([^\]]+)\]", token)
            if m:
                # TOUCH_HOLD with bracket
                hold_sec = _length_to_seconds(m.group(1), bpm)
                return Note(
                    type=NoteType.TOUCH_HOLD,
                    data=Touch_data(Touch_area=area, isFirework=has_f, holdTime=hold_sec),
                )
            else:
                # pseudo-touch-hold (no bracket): treat as TOUCH
                return Note(
                    type=NoteType.TOUCH,
                    data=Touch_data(Touch_area=area, isFirework=has_f),
                )
        else:
            return Note(
                type=NoteType.TOUCH,
                data=Touch_data(Touch_area=area, isFirework=has_f),
            )

    # ── slide parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _is_slide(token: str) -> bool:
        for s in _SHAPE_TOKENS:
            if s in token:
                return True
        return False

    def _parse_slide(self, token: str, bpm: float) -> Note:
        """
        Parse slide token into Note(type=SLIDE, data=list[SlideSegment]).

        Handles:
          basic:    1-4[8:3]
          arc:      1>2[4:3]
          grand-v:  1V35[2:1]
          chaining: 1-4q7-2[1:2]  or  1-4[2:1]q7[2:1]-2[1:1]
          multiple: 1-4[4:3]*-6[8:5]
          no-head:  1?-5[2:1]  /  1!-5[2:1]
          break:    1-4[8:3]b
          force-star: 1$-4[8:3]
          no-star:    1@-4[8:3]
          fake-rotate: 1$$-4[8:3]
        """
        t = token

        # trailing 'b' means entire slide is break
        is_slide_break = t.endswith("b") and not t.endswith("hb")
        if is_slide_break:
            t = t[:-1]

        # no-head markers at the very start
        is_no_head = False
        if t and t[0] in ("?", "!"):
            is_no_head = True
            t = t[1:]

        segments, is_ex, is_slide_break = self._build_slide_segments(t, bpm, is_slide_break, is_no_head)
        return Note(type=NoteType.SLIDE, data=segments, isBreak=is_slide_break, isEx=is_ex)

    def _build_slide_segments(
        self, t: str, bpm: float, is_slide_break: bool, is_no_head: bool
    ) -> tuple[list[SlideSegment], bool, bool]:
        """Walk the slide token string and build a list of SlideSegment."""
        segments: list[SlideSegment] = []
        i = 0
        n = len(t)

        # parse start lane (possibly with $, @, $$, x)
        if i >= n or t[i] not in "12345678":
            return segments, False, is_slide_break

        start_lane = _lane(t[i])
        i += 1

        # consume modifiers between lane and shape: @, $, $$, x
        is_force_star = False
        is_no_star = False
        is_fake_rotate = False
        is_ex = False
        while i < n and t[i] in ("@", "$", "x", "b", "?", "!"):
            if t[i] == "$":
                if i + 1 < n and t[i + 1] == "$":
                    is_fake_rotate = True
                    i += 2
                else:
                    is_force_star = True
                    i += 1
            elif t[i] == "@":
                is_no_star = True
                i += 1
            elif t[i] == "x":
                is_ex = True
                i += 1
            elif t[i] == "b":
                # 'b' before shape = break (some charts write it here)
                i += 1  # just consume, isSlideBreak handled outside
            elif t[i] in ("?", "!"):
                is_no_head = True
                i += 1

        # '*' separator between multiple slides at same start
        # we handle it in the outer loop

        cur_start = start_lane

        while i < n:
            # skip '*' separator (multiple slide)
            if t[i] == "*":
                i += 1
                # reset cur_start to original start for multiple slides
                # (multiple slides share the same start point)
                # actually after *, next segment starts from the *original* start
                cur_start = start_lane

            # try to match a shape
            shape = self._match_shape(t, i)
            if shape is None:
                break
            # determine clockwise for Circle shapes
            is_cw = None
            if shape == SlideShape.Circle:
                if t[i] == ">":
                    is_cw = True
                elif t[i] == "<":
                    is_cw = False
                # else '^' -> None (auto)
            i += len(shape.name) if shape in (
                SlideShape.PP, SlideShape.QQ
            ) else self._shape_char_len(shape, t, i)

            # For GrandV: middle_lane is an extra digit before end lane
            middle_lane = None
            if shape == SlideShape.GrandV:
                if i < n and t[i] in "12345678":
                    middle_lane = _lane(t[i])
                    i += 1

            # end lane
            if i >= n or t[i] not in "12345678":
                break
            end_lane = _lane(t[i])
            i += 1

            # optional 'b' break marker before bracket
            if i < n and t[i] == "b":
                is_slide_break = True
                i += 1

            # optional [wait##trace] for this segment
            wait_sec = 240.0 / bpm if bpm > 0 else 0.0
            trace_sec = 0.0
            if i < n and t[i] == "[":
                j = t.index("]", i)
                bracket_content = t[i + 1 : j]
                wait_sec, trace_sec = self._parse_slide_bracket(bracket_content, bpm)
                i = j + 1

            seg = SlideSegment(
                shape=shape,
                start_lane=cur_start,
                end_lane=end_lane,
                wait_duration=wait_sec,
                trace_duration=trace_sec,
                isClockwise=is_cw,
                middle_lane=middle_lane,
                isForceStar=is_force_star,
                isFakeRotate=is_fake_rotate,
                isSlideBreak=is_slide_break,
                isSlideNoHead=is_no_star or is_no_head,
            )
            segments.append(seg)
            cur_start = end_lane  # chaining: next starts where this ended

            # check for clockwise indicator on arc
            if shape == SlideShape.Circle and len(segments) > 0:
                # direction already encoded in shape usage, no extra field needed
                pass

        return segments, is_ex, is_slide_break

    @staticmethod
    def _match_shape(t: str, i: int) -> SlideShape | None:
        """Try to match a slide shape token starting at position i."""
        for tok in _SHAPE_TOKENS:
            end = i + len(tok)
            if t[i:end] == tok:
                return _SHAPE_MAP[tok]
        return None

    @staticmethod
    def _shape_char_len(shape: SlideShape, t: str, i: int) -> int:
        """Return the character length of the shape token at position i."""
        if shape in (SlideShape.PP, SlideShape.QQ):
            return 2
        return 1

    def _parse_slide_bracket(self, content: str, bpm: float) -> tuple[float, float]:
        """
        Parse slide bracket content into (wait_seconds, trace_seconds).

        Formats:
          [8:3]               -> wait=1 beat @ bpm, trace = (240/bpm/8)*3
          [160#8:3]           -> wait=1 beat @ 160, trace = (240/160/8)*3
          [160#2]             -> wait=1 beat @ 160, trace = 2 seconds
          [3##1.5]            -> wait=3 sec, trace=1.5 sec
          [3##8:3]            -> wait=3 sec, trace=(240/bpm/8)*3
          [3##160#8:3]        -> wait=3 sec, trace=(240/160/8)*3
        """
        if "##" in content:
            parts = content.split("##", 1)
            wait_sec = float(parts[0]) if parts[0] else 0.0
            trace_rest = parts[1]
        else:
            # no explicit wait: 1 beat at current or specified BPM
            if "#" in content and ":" not in content and content.count("#") == 1:
                # format: BPM#seconds  e.g. 160#2
                bp_s, sec_s = content.split("#", 1)
                trace_bpm = float(bp_s)
                return 240.0 / trace_bpm, float(sec_s)

            if "#" in content:
                # BPM#divider:mult
                bp_s, rest = content.split("#", 1)
                trace_bpm = float(bp_s)
                return 240.0 / trace_bpm, _length_to_seconds(rest, trace_bpm)

            # plain divider:mult  -> wait = 1 beat at bpm
            wait_sec = 240.0 / bpm if bpm > 0 else 0.0
            return wait_sec, _length_to_seconds(content, bpm)

        # trace_rest: can be '#seconds', 'BPM#divider:mult', or 'divider:mult'
        if trace_rest.startswith("#"):
            return wait_sec, float(trace_rest[1:])

        if "#" in trace_rest:
            bp_s, rest = trace_rest.split("#", 1)
            trace_bpm = float(bp_s)
            return wait_sec, _length_to_seconds(rest, trace_bpm)

        return wait_sec, _length_to_seconds(trace_rest, bpm)

    # ── BPM / length divider tracking ─────────────────────────────────────

    def _parse_current_time_per_comma(
        self,
        note: str,
        current_bpm: float,
        current_length_divider: float,
        current_per_comma_length: float,
    ) -> tuple[float, float, float]:
        bpm_match = re.search(r"\(([^)]*)\)", note)
        length_match = re.search(r"\{([^}]*)\}", note)

        next_bpm = current_bpm
        next_divider = current_length_divider
        next_per_comma = current_per_comma_length

        if bpm_match:
            next_bpm = float(bpm_match.group(1))

        if length_match:
            raw_value = length_match.group(1).strip()
            if raw_value.startswith("#"):
                next_divider = 0.0
                next_per_comma = float(raw_value[1:])
            else:
                next_divider = float(raw_value)
                if next_bpm > 0:
                    next_per_comma = 240.0 / next_bpm / next_divider
        elif bpm_match:
            # BPM changed but no length change → recalculate with existing divider
            if next_bpm > 0 and next_divider > 0:
                next_per_comma = 240.0 / next_bpm / next_divider

        if bpm_match or length_match:
            return next_bpm, next_divider, next_per_comma

        if next_bpm > 0 and next_divider > 0:
            next_per_comma = 240.0 / next_bpm / next_divider

        return next_bpm, next_divider, next_per_comma

    # ── main parse ────────────────────────────────────────────────────────

    def parse(self, text: str, music_data: list[dict] | None = None):
        self.chart = Chart(all_levels=[None] * 7)

        title_match = re.search(r"&title=([^\n]+)", text)
        if title_match:
            self.chart.title = title_match.group(1).strip()

        artist_match = re.search(r"&artist=([^\n]+)", text)
        if artist_match:
            self.chart.artist = artist_match.group(1).strip()

        # 调用方显式传入本地歌曲表时才使用云端元数据；解析本身不读文件或联网。
        song = _match_music(text, self.chart.title, music_data) if music_data is not None else None

        for level in range(0, 7):
            level_match = re.search(
                rf"&inote_{level}=([\s\S]*?)(?=&lv_[0-9]|&inote_[0-9]|$)",
                text,
            )
            if not level_match:
                continue

            level_content = level_match.group(1)
            chart_content = level_content.split(",")
            chart_content_strip = [item.strip() for item in chart_content]

            current_bpm = 0.0
            current_length_divider = 0.0
            current_per_comma_length = 1.0
            self.current_time = 0.0
            frames_by_time: dict[float, list[Note]] = {}

            for raw_note in chart_content_strip:
                # strip comments (|| to end of line)
                if "||" in raw_note:
                    raw_note = raw_note[:raw_note.index("||")]
                if not raw_note:
                    self.current_time += current_per_comma_length
                    continue
                if raw_note == "E":
                    break

                current_bpm, current_length_divider, current_per_comma_length = (
                    self._parse_current_time_per_comma(
                        raw_note,
                        current_bpm,
                        current_length_divider,
                        current_per_comma_length,
                    )
                )

                cleaned = re.sub(r"\([^)]*\)", "", raw_note)
                cleaned = re.sub(r"\{[^}]*\}", "", cleaned).strip()
                if not cleaned or cleaned == "E":
                    self.current_time += current_per_comma_length
                    continue

                notes = self._parse_note(cleaned, current_bpm)
                if notes is None:
                    self.current_time += current_per_comma_length
                    continue

                frames_by_time.setdefault(self.current_time, []).extend(notes)
                self.current_time += current_per_comma_length

            frames = [
                Frame(
                    notes=tuple(notes),
                    time_sec=t,
                )
                for t, notes in sorted(frames_by_time.items())
            ]
            level_name = f"level_{level + 1}"
            level_query_match = re.search(rf"&lv_{level}=([^\n]+)", text)
            level_query = _level_to_float(level_query_match.group(1).strip()) if level_query_match else None
            ds_index = level - 2  # maidata lv_2..lv_6 对应 API Basic..Re:Master。
            if song is not None and 0 <= ds_index < len(song.get("ds", [])):
                try:
                    level_query = float(song["ds"][ds_index])
                except (TypeError, ValueError):
                    pass
            self.chart.all_levels[level] = Level(
                level_name=level_name, level_query=level_query, frames=frames
            )

        return self.chart

    # ── eval ──────────────────────────────────────────────────────────────

    @staticmethod
    def _update_extremes(extremes: dict[str, dict[str, int]], key: str, value: int) -> None:
        slot = extremes.get(key)
        if slot is None:
            extremes[key] = {"min": value, "max": value}
            return
        if value < slot["min"]:
            slot["min"] = value
        if value > slot["max"]:
            slot["max"] = value

    def _eval_collect_note(self, note: Note, extremes: dict[str, dict[str, int]]) -> None:
        t = note.type
        if t == NoteType.SLIDE:
            for seg in note.data:
                self._update_extremes(extremes, "wait_duration", seg.wait_duration)
                self._update_extremes(extremes, "trace_duration", seg.trace_duration)
        elif t == NoteType.HOLD:
            self._update_extremes(extremes, "holdTime", note.data.holdTime)
        elif t == NoteType.TOUCH_HOLD:
            if note.data.holdTime is not None:
                self._update_extremes(extremes, "holdTime", note.data.holdTime)

    def eval(self, text: str) -> dict[str, dict[str, int]]:
        chart = self.parse(text)
        extremes: dict[str, dict[str, int]] = {}
        for level in chart.all_levels:
            if level is None:
                continue
            prev_time: float | None = None
            for frame in level.frames:
                for note in frame.notes:
                    self._eval_collect_note(note, extremes)
                if prev_time is not None:
                    gap_ms = int(round((frame.time_sec - prev_time) * 1000))
                    self._update_extremes(extremes, "frame_gap_ms", gap_ms)
                prev_time = frame.time_sec
        return extremes

    # ── Token encoding helpers ───────────────────────────────────────────

    @staticmethod
    def _ts_token(seconds: float) -> int:
        from tokenizer import seconds_to_token

        return seconds_to_token(seconds)

    @staticmethod
    def _lane_token(lane: TapType) -> int:
        return LANE_BASE + lane.value - 1

    @staticmethod
    def _touch_token(touch: TouchType) -> int:
        return TOUCH_BASE + touch.value - 1

    @staticmethod
    def _shape_token(shape: SlideShape) -> int:
        return SLIDE_SHAPE_BASE + shape.value - 1

    def _encode_note_tokens(self, note: Note) -> list[int]:
        from tokenizer import encode_note

        return encode_note(note)

    # ── tensor export (token vocabulary) ──────────────────────────────────

    @staticmethod
    def _note_end_sec(frame_time: float, note: "Note") -> float:
        """Absolute end-time of a single note.

        For TAP: same as frame_time.
        For HOLD / TOUCH_HOLD: frame_time + holdTime.
        For SLIDE: frame_time + max(wait + trace) across segments.
        """
        t = note.type
        if t == NoteType.HOLD:
            return frame_time + (note.data.holdTime or 0.0)
        elif t == NoteType.TOUCH_HOLD:
            return frame_time + (note.data.holdTime or 0.0)
        elif t == NoteType.SLIDE:
            end = frame_time
            for seg in note.data:
                candidate = frame_time + seg.wait_duration + seg.trace_duration
                if candidate > end:
                    end = candidate
            return end
        return frame_time

    def to_tensor(self, level_idx: int = 5) -> tuple[list[float], list[torch.Tensor]]:
        """Export a single level to segmented token sequences.

        Follows doc/token化设计.md §6:
          - Segments at token granularity: split only after a token has fully
            ended (hold/slide finish time), never mid-token.
          - Each segment ≤ 30 seconds.
          - Must track the **global** active-note latest end (e.g. a HOLD
            spanning many frames), not just the current frame's end.
          - TS tokens within a segment are relative (segment_offset subtracted).
          - Frames without notes are skipped.

        Returns:
            (offsets, tensors) where len(offsets) == len(tensors).
            offsets[i] = absolute time (seconds) of segment i's start.
            tensors[i] = 1-D int64 token sequence for segment i.
            Empty ([], []) if the level is missing.
        """
        if not (0 <= level_idx < len(self.chart.all_levels)):
            raise ValueError(f"level_idx {level_idx} out of range")
        level = self.chart.all_levels[level_idx]
        if level is None:
            return ([], [])

        # ── build token groups with float timestamps ──
        # Each group: (note_tokens, frame_time, frame_end_time)
        groups: list[tuple[list[int], float, float]] = []

        for frame in level.frames:
            if not frame.notes:
                continue
            toks: list[int] = []
            frame_end = frame.time_sec
            for note in frame.notes:
                toks.extend(self._encode_note_tokens(note))
                note_end = self._note_end_sec(frame.time_sec, note)
                if note_end > frame_end:
                    frame_end = note_end
            groups.append((toks, frame.time_sec, frame_end))

        if not groups:
            return ([], [])

        # ── dynamic segmentation (§6.2) ──
        # 硬约束：每段 ≤ 30 秒（TS_0~TS_2999 @ 100fps = 30s）。
        # 不按固定 30 秒硬切，在 token 边界处切割，不破坏任何音符。
        # 关键：seg_max_end 跟踪当前段内**所有**活跃音符的最晚结束时间
        #（例如一个 HOLD 跨越多个帧，期间有其他音符），不能只看当前帧的 end。
        # 先判断再加入，确保每段 ≤ 30s。
        # 第一段从音频 0 秒开始，保留首个音符前的静音上下文。
        # 后续段仍按谱面 frame 起点动态切分，避免训练/原始音频推理起点不一致。
        seg_offset = 0.0
        seg_max_end = groups[0][2]
        current_seg: list[tuple[list[int], float]] = []
        all_segs: list[tuple[float, list[tuple[list[int], float]]]] = []

        for note_toks, start, end in groups:
            # 先判断：加入这个 group 后，全局最晚 end 是否超过 30s？
            candidate_max = max(seg_max_end, end)
            if current_seg and (candidate_max - seg_offset > 30.0):
                # 超过了，先 flush 当前段（当前 group 不加入）
                all_segs.append((seg_offset, current_seg))
                # 新段从当前帧的 start 开始，保证相对时间 ≥ 0
                seg_offset = start
                seg_max_end = end
                current_seg = [(note_toks, start)]
            else:
                seg_max_end = candidate_max
                current_seg.append((note_toks, start))

        if current_seg:
            all_segs.append((seg_offset, current_seg))

        if not all_segs:
            return ([], [])

        # ── build final token lists with relative TS (§6.1) ──
        offsets: list[float] = []
        tensors: list[torch.Tensor] = []
        for abs_offset, seg in all_segs:
            tokens: list[int] = [SOS]
            for note_toks, frame_time in seg:
                tokens.append(FRAME_START)
                tokens.append(self._ts_token(frame_time - abs_offset))
                tokens.extend(note_toks)
                tokens.append(FRAME_END)
            tokens.append(EOS)
            offsets.append(abs_offset)
            tensors.append(torch.tensor(tokens, dtype=torch.int64))

        return (offsets, tensors)

    # ── §6 Data Loader utilities ───────────────────────────────────────────

    @staticmethod
    def fit_tokens(token_list: list[int], max_len: int) -> torch.Tensor:
        """Pad or truncate a token list to exactly max_len.

        §6.4: 由于每段的 token 数量不等，需要填充到统一长度。
        模型通过 attention mask 忽略 PAD 位置。
        """
        if len(token_list) >= max_len:
            return torch.tensor(token_list[:max_len], dtype=torch.int64)
        padded = token_list + [PAD] * (max_len - len(token_list))
        return torch.tensor(padded, dtype=torch.int64)

    def extract_time_slots(self, tokens: list[int]) -> list[float]:
        """Extract relative time (seconds) for each token in a segment.

        §6.5: 每个 token 的时间位置 = segment_offset + token_relative_time。
        返回的 list 与 tokens 等长，非 TS 位置返回前一个 TS 的值。
        """
        slots: list[float] = []
        current_time = 0.0
        for tok in tokens:
            if TS_BASE <= tok < TS_BASE + 3000:
                current_time = (tok - TS_BASE) / 100.0
            slots.append(current_time)
        return slots

    def to_training_data(
        self,
        level_idx: int = 5,
        max_tokens: int = 0,
    ) -> tuple[list[float], torch.Tensor, torch.Tensor]:
        """Export a level ready for the Data Loader.

        §6.3 Data Loader 工作流程:
          1. 按动态分段策略将 token 序列切成 N 段
          2. 每段记录 segment_offset + tokens
          3. 填充到统一长度
          4. 生成 attention mask

        Returns:
            offsets: list[float]  — 每段的绝对起始时间（秒）
            padded: (N, max_tokens) int64  — 填充后的 token 张量
            mask:   (N, max_tokens) bool   — True = 有效 token，False = PAD
        """
        offsets, tensors = self.to_tensor(level_idx)
        if not offsets:
            return ([], torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool))

        token_lists = [t.tolist() for t in tensors]

        if max_tokens <= 0:
            max_tokens = max(len(tl) for tl in token_lists)

        padded_rows = []
        mask_rows = []
        for tl in token_lists:
            length = min(len(tl), max_tokens)
            padded_rows.append(self.fit_tokens(tl, max_tokens))
            mask_rows.append(
                torch.cat([torch.ones(length, dtype=torch.bool),
                           torch.zeros(max_tokens - length, dtype=torch.bool)])
            )

        padded = torch.stack(padded_rows)
        mask = torch.stack(mask_rows)
        return (offsets, padded, mask)

    def get_audio_slice(
        self,
        mel: torch.Tensor,
        segment_offset: float,
        segment_duration: float,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
    ) -> torch.Tensor:
        """Slice mel spectrogram for a segment's time window.

        §6.5: 用 segment_offset 和 segment_duration 从 mel 中切出对应帧。

        Args:
            mel: (n_mels, T_total) 完整音频的 mel spectrogram
            segment_offset: 段的绝对起始时间（秒）
            segment_duration: 段的持续时间（秒）
            sample_rate: 音频采样率
            hop_length: mel 的 hop length
            n_mels: mel 频带数

        Returns:
            (n_mels, T_segment) 切片后的 mel
        """
        frames_per_sec = sample_rate / hop_length
        start_frame = int(segment_offset * frames_per_sec)
        end_frame = int((segment_offset + segment_duration) * frames_per_sec)
        start_frame = max(0, min(start_frame, mel.shape[1]))
        end_frame = max(start_frame, min(end_frame, mel.shape[1]))
        return mel[:, start_frame:end_frame]

    # ── tensor import (token vocabulary) ────────────────────────────────────

    def _parse_token_segment(self, tok: list[int]) -> list[Frame]:
        """Parse a flat token list (one segment) into Frame objects.

        Returns a list of Frame with relative time_sec (caller adds offset).
        """
        from tokenizer import decode_frames

        return decode_frames(tok)

    def parse_from_tensor(
        self,
        data: tuple[list[float], list[torch.Tensor]],
        level_idx: int = 5,
        title: str = "generated",
        artist: str | None = None,
        level_query: float | None = None,
    ) -> "compiler":
        """Import token data into the compiler's chart at level_idx.

        Accepts:
          - (offsets, tensors) tuple from to_tensor() (preferred, §6 format).
          - 2-D tensor (num_segments, max_tokens): backward compat, offsets
            inferred by accumulating last-frame relative times.

        Inverse of to_tensor().  Returns self for chaining.
        """
        if isinstance(data, tuple):
            offsets, tensors = data
            if len(offsets) != len(tensors):
                print(f"[parse_from_tensor] 警告: offsets 长度 ({len(offsets)}) != tensors 长度 ({len(tensors)})")
            segment_rows = [t.tolist() for t in tensors]
        elif isinstance(data, torch.Tensor):
            if data.ndim == 2:
                segment_rows = [data[i].tolist() for i in range(data.shape[0])]
            elif data.ndim == 1:
                segment_rows = [data.tolist()]
            else:
                raise ValueError(f"Expected 1-D or 2-D tensor, got shape {data.shape}")
            offsets = None  # will be inferred
        else:
            raise TypeError(f"Expected tuple or Tensor, got {type(data)}")

        if not (0 <= level_idx < 7):
            raise ValueError(f"level_idx {level_idx} out of range 0..6")

        all_frames: list[Frame] = []
        inferred_offset = 0.0
        skipped_empty = 0
        skipped_bad_sos = 0

        for seg_idx, seg_tok in enumerate(segment_rows):
            # strip PAD tokens from the end
            while seg_tok and seg_tok[-1] == PAD:
                seg_tok.pop()
            if not seg_tok:
                skipped_empty += 1
                continue

            # 校验 SOS 开头
            if seg_tok[0] != SOS:
                print(f"[parse_from_tensor] 警告: 段 {seg_idx} 不以 SOS 开头 (实际={seg_tok[0]})")
                skipped_bad_sos += 1

            # use explicit offset if available, else infer
            if offsets is not None:
                abs_offset = offsets[seg_idx]
                if abs_offset < 0:
                    print(f"[parse_from_tensor] 警告: 段 {seg_idx} 偏移为负 ({abs_offset:.3f}s)")
            else:
                abs_offset = inferred_offset

            rel_frames = self._parse_token_segment([t for t in seg_tok if t not in (SOS, EOS, PAD)])
            if not rel_frames:
                print(f"[parse_from_tensor] 段 {seg_idx} (offset={abs_offset:.3f}s) 解码出 0 帧")

            for f in rel_frames:
                all_frames.append(Frame(
                    notes=f.notes,
                    time_sec=f.time_sec + abs_offset,
                ))

            # for inferred offsets: next segment starts after last frame
            if offsets is None and rel_frames:
                inferred_offset = abs_offset + rel_frames[-1].time_sec

        all_frames.sort(key=lambda f: f.time_sec)

        # 汇总日志
        total_segs = len(segment_rows)
        decoded_segs = total_segs - skipped_empty
        print(
            f"[parse_from_tensor] 完成: {len(all_frames)} 帧, "
            f"{decoded_segs}/{total_segs} 段有效"
            f"{f', 跳过 {skipped_empty} 空段' if skipped_empty else ''}"
            f"{f', {skipped_bad_sos} 段缺 SOS' if skipped_bad_sos else ''}"
        )

        # Preserve existing chart metadata
        if self.chart is not None:
            chart_title = self.chart.title if title == "generated" else title
            chart_artist = self.chart.artist if artist is None else artist
        else:
            chart_title = title
            chart_artist = artist or "default"

        if self.chart is None or self.chart.all_levels == []:
            self.chart = Chart(all_levels=[None] * 7, title=chart_title, artist=chart_artist)
        else:
            self.chart.title = chart_title
            self.chart.artist = chart_artist
            if len(self.chart.all_levels) <= level_idx:
                self.chart.all_levels.extend([None] * (level_idx + 1 - len(self.chart.all_levels)))

        self.chart.all_levels[level_idx] = Level(
            level_name=f"level_{level_idx + 1}",
            level_query=level_query if level_query is not None else 0.0,
            frames=all_frames,
        )

        print(f"[parse_from_tensor] level {level_idx}: {len(all_frames)} frames decoded")
        return self

    # ── note-to-text reconstruction ───────────────────────────────────────

    # TapType value -> lane number string
    _LANE_NUM = {
        TapType.LANE1: "1", TapType.LANE2: "2", TapType.LANE3: "3", TapType.LANE4: "4",
        TapType.LANE5: "5", TapType.LANE6: "6", TapType.LANE7: "7", TapType.LANE8: "8",
    }

    # SlideShape + isClockwise -> shape character(s)
    _SHAPE_CHAR = {
        (SlideShape.Line, None): "-",
        (SlideShape.Circle, True): ">",
        (SlideShape.Circle, False): "<",
        (SlideShape.Circle, None): "^",
        (SlideShape.V, None): "v",
        (SlideShape.GrandV, None): "V",
        (SlideShape.P, None): "p",
        (SlideShape.Q, None): "q",
        (SlideShape.PP, None): "pp",
        (SlideShape.QQ, None): "qq",
        (SlideShape.S, None): "s",
        (SlideShape.Z, None): "z",
        (SlideShape.Wifi, None): "w",
    }

    @staticmethod
    def _beats_to_divider_mult(beats: float) -> tuple[int, int] | None:
        """Approximate a beat count as (divider, mult) with mult/divider ≈ beats.
        Returns None if no clean approximation found."""
        if beats <= 0:
            return None
        for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]:
            m = beats * d
            mi = int(round(m))
            if mi > 0 and abs(m - mi) / max(mi, 1) < 0.02:
                return d, mi
        return None

    def _duration_to_notation(self, seconds: float, bpm: float) -> str:
        """Convert duration in seconds to simai [X:Y] notation at given BPM.
        Tries power-of-2 denominators, falls back to computed BPM with [BPM#D:M]."""
        if seconds <= 0:
            return "1:1"
        beats = seconds * bpm / 240.0
        for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]:
            y = beats * x
            yi = int(round(y))
            if yi > 0 and abs(y - yi) / max(yi, 1) < 0.02:
                return f"{x}:{yi}"
        # fallback: compute a BPM where this duration = 1 beat, find clean D:M
        implied_bpm = 240.0 / seconds
        result = self._beats_to_divider_mult(beats)
        if result is not None:
            d, m = result
            bpm_str = f"{implied_bpm:.4f}"
            bpm_str = bpm_str.rstrip('0').rstrip('.')
            return f"{bpm_str}#{d}:{m}"
        # last resort: #seconds
        s = f"{seconds:.6f}"
        s = s.rstrip('0').rstrip('.')
        return f"#{s}"

    def _note_to_text(self, note: Note, bpm: float) -> str:
        """Reconstruct simai text for a single Note at the given BPM."""
        t = note.type

        if t == NoteType.TAP:
            s = self._LANE_NUM[note.data]
            if note.isBreak:
                s += "b"
            if note.isEx:
                s += "x"
            return s

        if t == NoteType.HOLD:
            s = self._LANE_NUM[note.data.lane]
            if note.isBreak:
                s += "b"
            if note.isEx:
                s += "x"
            s += "h"
            hold_sec = note.data.holdTime
            if hold_sec > 0:
                s += f"[{self._duration_to_notation(hold_sec, bpm)}]"
            return s

        if t == NoteType.TOUCH:
            s = note.data.Touch_area.name
            if note.data.isFirework:
                s += "f"
            return s

        if t == NoteType.TOUCH_HOLD:
            s = note.data.Touch_area.name
            if note.data.isFirework:
                s += "f"
            hold_sec = note.data.holdTime
            if hold_sec > 0:
                s += f"h[{self._duration_to_notation(hold_sec, bpm)}]"
            else:
                s += "h"
            return s

        if t == NoteType.SLIDE:
            return self._slide_to_text(note, bpm)

        return "?"

    def _slide_to_text(self, note: Note, bpm: float) -> str:
        """Reconstruct simai text for a SLIDE note."""
        segments: list[SlideSegment] = note.data
        if not segments:
            return "?"

        parts: list[str] = []
        prev_end = None
        first_start = segments[0].start_lane if segments else None

        for i, seg in enumerate(segments):
            is_first = (i == 0)

            # shape character
            shape_key = (seg.shape, seg.isClockwise)
            shape_char = self._SHAPE_CHAR.get(shape_key, "-")

            # start lane / separator
            if is_first:
                start_num = self._LANE_NUM[seg.start_lane]
                mod = ""
                if note.isEx:
                    mod += "x"
                if seg.isSlideNoHead:
                    mod += "?"
                if seg.isForceStar:
                    mod += "$"
                if seg.isFakeRotate:
                    mod += "$$"
                parts.append(start_num + mod)
            elif seg.start_lane == prev_end:
                pass  # chaining: shape follows directly
            elif seg.start_lane == first_start:
                parts.append("*")  # multiple slide: same start, use *
            else:
                # different start (shouldn't happen normally)
                parts.append(self._LANE_NUM[seg.start_lane])

            # shape + middle lane (GrandV) + end lane
            end_num = self._LANE_NUM[seg.end_lane]
            mid = ""
            if seg.middle_lane is not None:
                mid = self._LANE_NUM[seg.middle_lane]
            parts.append(shape_char + mid + end_num)

            # duration bracket (only if trace > 0)
            if seg.trace_duration > 0:
                trace_sec = seg.trace_duration
                wait_sec = seg.wait_duration
                default_wait = 240.0 / bpm  # 1 beat at generation BPM
                if abs(wait_sec - default_wait) < 0.005:
                    # wait = 1 beat at gen BPM, use simple [D:M] notation
                    trace_str = self._duration_to_notation(trace_sec, bpm)
                    parts.append(f"[{trace_str}]")
                else:
                    # wait ≠ 1 beat at gen BPM: find BPM where wait = 1 beat,
                    # then express trace at that same BPM.
                    wait_bpm = 240.0 / wait_sec
                    trace_beats = trace_sec * wait_bpm / 240.0
                    # try to find clean D:M for trace at wait_bpm
                    dm = self._beats_to_divider_mult(trace_beats)
                    if dm is not None:
                        d, m = dm
                        bpm_s = f"{wait_bpm:.4f}".rstrip('0').rstrip('.')
                        parts.append(f"[{bpm_s}#{d}:{m}]")
                    else:
                        # trace can't be expressed as clean beats; use seconds
                        ts = f"{trace_sec:.6f}".rstrip('0').rstrip('.')
                        bpm_s = f"{wait_bpm:.4f}".rstrip('0').rstrip('.')
                        parts.append(f"[{bpm_s}#{ts}]")

            prev_end = seg.end_lane

        # join parts, add break marker at end
        result = "".join(parts)
        if segments[-1].isSlideBreak:
            result += "b"
        return result

    # ── generate ──────────────────────────────────────────────────────────

    def generate(self) -> str:
        lines: list[str] = []
        lines.append(f"&title={self.chart.title}")
        lines.append(f"&artist={self.chart.artist}")
        lines.append("&first=0")

        gen_bpm = 12000.0
        gen_divider = 1.0
        per_comma_time = 240.0 / gen_bpm / gen_divider  # 0.02 seconds

        for level_idx, level in enumerate(self.chart.all_levels):
            if level is None:
                continue

            lines.append(f"&lv_{level_idx}={level.level_query or 0}")
            lines.append(f"&inote_{level_idx}=")

            if not level.frames:
                lines.append("E")
                continue

            output_lines: list[str] = []
            commas_emitted = 0
            first_note = True

            for frame in sorted(level.frames, key=lambda f: f.time_sec):
                if not frame.notes:
                    continue

                # reconstruct note text from parsed Note objects
                combined = "/".join(self._note_to_text(n, gen_bpm) for n in frame.notes)

                # derive comma count from time_sec
                target_commas = max(0, int(round(frame.time_sec / per_comma_time)))
                gap_commas = max(0, target_commas - commas_emitted)

                if first_note:
                    prefix = "(12000){1}" + "," * gap_commas
                    output_lines.append(f"{prefix}{combined},")
                    first_note = False
                else:
                    if gap_commas > 0 and output_lines:
                        output_lines[-1] += "," * gap_commas
                    output_lines.append(f"{combined},")

                commas_emitted = target_commas + 1

            lines.extend(output_lines)
            lines.append("E")

        return "\n".join(lines)


# ── batch tool ────────────────────────────────────────────────────────────

def batch_main():
    import difflib

    charts_dir = Path(__file__).resolve().parent.parent / "charts"
    tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
    flow_a_dir = tmp_dir / "txt2txt"
    flow_b_dir = tmp_dir / "txt2ternsor2txt"
    found = errors = 0
    diffs = 0
    overall: dict[str, dict[str, int]] = {}
    diff_files: list[str] = []
    for chart_dir, _dirs, files in charts_dir.walk():
        if "maidata.txt" not in files:
            continue
        rel = chart_dir.relative_to(charts_dir)
        path_a = flow_a_dir / rel / "maidata.txt"
        path_b = flow_b_dir / rel / "maidata.txt"
        try:
            text = (chart_dir / "maidata.txt").read_text(encoding="utf-8")
            parser = compiler(hop_length=512, sample_rate=44100)
            extremes = parser.eval(text)
            path_a.parent.mkdir(parents=True, exist_ok=True)
            path_b.parent.mkdir(parents=True, exist_ok=True)
            found += 1
            for key, slot in extremes.items():
                cur = overall.get(key)
                if cur is None:
                    overall[key] = dict(slot)
                else:
                    if slot["min"] < cur["min"]:
                        cur["min"] = slot["min"]
                    if slot["max"] > cur["max"]:
                        cur["max"] = slot["max"]

            # ── flow (a): parse → generate ──
            title = parser.chart.title
            artist = parser.chart.artist
            text_a = parser.generate()
            path_a.write_text(text_a, encoding="utf-8")

            # ── flow (b): parse → tensor → parse_from_tensor → generate ──
            tensors: dict[int, tuple[list[float], list[torch.Tensor], float]] = {}
            for li in range(7):
                lv = parser.chart.all_levels[li]
                if lv is not None:
                    offsets, segs = parser.to_tensor(level_idx=li)
                    tensors[li] = (offsets, segs, lv.level_query)
            parser.chart = None
            for li, (offsets, segs, lq) in tensors.items():
                parser.parse_from_tensor((offsets, segs), level_idx=li, title=title, artist=artist, level_query=lq)
            text_b = parser.generate()
            path_b.write_text(text_b, encoding="utf-8")

            # 对比
            if text_a != text_b:
                diffs += 1
                diff_files.append(str(rel))

        except Exception as e:
            import traceback
            errors += 1
            print(f"[ERR] {rel}: {e}")
            traceback.print_exc()

    # 写出 diff 文件列表供子代理分析
    diff_list_path = Path(__file__).resolve().parent.parent / ".openclaw" / "tmp" / "diff_files.txt"
    diff_list_path.parent.mkdir(parents=True, exist_ok=True)
    diff_list_path.write_text("\n".join(diff_files), encoding="utf-8")

    print(f"\nDone: {found} parsed, {errors} errors, {diffs} diffs")
    print(f"Diff file list: {diff_list_path}")
    print(f"Flow A dir: {flow_a_dir}")
    print(f"Flow B dir: {flow_b_dir}")
    print(f"Overall extremes per field: {overall}")


def _self_check() -> None:
    from unittest.mock import patch

    from chart_cache import _config, _scan_sources

    chart = compiler().parse("&title=test\n&shortid=1\n&lv_2=13\n&lv_3=13+\n&inote_2=E\n&inote_3=E")
    assert chart.all_levels[2].level_query == 13.0
    assert chart.all_levels[3].level_query == 13.5

    song = {"id": "1", "ds": [2.3, 5.7], "basic_info": {"genre": "宴会場"}}
    chart = compiler().parse(
        "&title=test\n&shortid=1\n&lv_2=1\n&lv_3=2\n&inote_2=E\n&inote_3=E",
        music_data=[song],
    )
    assert chart.all_levels[2].level_query == 2.3
    assert chart.all_levels[3].level_query == 5.7

    song = {"id": "1", "title": "test", "level": ["13+"], "ds": [13.7]}
    assert compiler().parse("&title=test\n&lv_2=13+\n&inote_2=E", music_data=[song]).all_levels[2].level_query == 13.7
    assert compiler().parse("&title=test\n&shortid=999\n&lv_2=13+\n&inote_2=E", music_data=[song]).all_levels[2].level_query == 13.7

    different_level = {"id": "1", "title": "test", "level": ["14"], "ds": [14.0]}
    assert compiler().parse(
        "&title=test\n&lv_2=13+\n&inote_2=E", music_data=[different_level],
    ).all_levels[2].level_query == 13.5
    assert compiler().parse("&title=test\n&lv_2=13+\n&inote_2=E").all_levels[2].level_query == 13.5

    common = (5, 22050, 256, 80, 1.0, 3000)
    assert _config(*common, "old") != _config(*common, "new")
    with patch("maidata_parser.load_music_data", side_effect=AssertionError):
        assert compiler().parse("&title=test\n&lv_2=13+\n&inote_2=E").all_levels[2].level_query == 13.5

    song = {"id": "1", "basic_info": {"genre": "宴会場"}}
    assert _match_music("&shortid=1", "test", [song]) is song
    with patch("chart_cache.compiler", side_effect=AssertionError):
        with patch("chart_cache.Path.rglob", return_value=[]):
            assert _scan_sources(Path("charts"), Path(".cache/charts")) == []
    print("[maidata-parser] 自检通过")


if __name__ == "__main__":
    import sys

    batch_main() if "--batch" in sys.argv else _self_check()
