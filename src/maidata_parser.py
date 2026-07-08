import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import torch


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
    wait_duration: int  # 等待时间帧数，默认一拍
    trace_duration: int  # 持续时间帧数

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
    holdTime: int | None = None  # 持续时间帧数，TouchHold 才有


@dataclass
class Hold_data:
    lane: TapType
    holdTime: int  # 持续时间帧数


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
    frame_idx: int
    notes: tuple[Note, ...] = ()
    time_sec: float = 0.0  # 精确时间（秒），用于生成器对齐


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


def _seconds_to_frames(seconds: float, time_per_frame: float) -> int:
    """Convert seconds to frame count."""
    if time_per_frame <= 0:
        return 0
    return max(1, int(round(seconds / time_per_frame)))


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


def _touch_hold_length(token: str, bpm: float, time_per_frame: float) -> int | None:
    """Extract hold length (frames) from a touch-hold token. Returns None for pseudo-hold."""
    m = re.search(r"\[([^\]]+)\]", token)
    if not m:
        return None  # pseudo-hold
    secs = _length_to_seconds(m.group(1), bpm)
    return _seconds_to_frames(secs, time_per_frame)


# ── compiler ──────────────────────────────────────────────────────────────

class compiler:
    def __init__(self, hop_length, sample_rate):
        self.chart = Chart(all_levels=[])
        self.time_pre_frame = hop_length / sample_rate
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
            secs = _length_to_seconds(m.group(1), bpm)
            hold_frames = _seconds_to_frames(secs, self.time_pre_frame)
        else:
            # pseudo-hold: treated as [1280:1]
            hold_frames = _seconds_to_frames(
                240.0 / bpm / 1280.0 if bpm > 0 else 0.0,
                self.time_pre_frame,
            )
        return Note(
            type=NoteType.HOLD,
            data=Hold_data(lane=lane, holdTime=hold_frames),
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
            # TOUCH_HOLD
            m = re.search(r"\[([^\]]+)\]", token)
            if m:
                secs = _length_to_seconds(m.group(1), bpm)
                hold_frames = _seconds_to_frames(secs, self.time_pre_frame)
            else:
                hold_frames = None  # pseudo-hold
            return Note(
                type=NoteType.TOUCH_HOLD,
                data=Touch_data(Touch_area=area, isFirework=has_f, holdTime=hold_frames),
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
            wait_frames = _seconds_to_frames(240.0 / bpm, self.time_pre_frame) if bpm > 0 else 0
            trace_frames = 0
            if i < n and t[i] == "[":
                j = t.index("]", i)
                bracket_content = t[i + 1 : j]
                wait_frames, trace_frames = self._parse_slide_bracket(bracket_content, bpm)
                i = j + 1

            seg = SlideSegment(
                shape=shape,
                start_lane=cur_start,
                end_lane=end_lane,
                wait_duration=wait_frames,
                trace_duration=trace_frames,
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

    def _parse_slide_bracket(self, content: str, bpm: float) -> tuple[int, int]:
        """
        Parse slide bracket content into (wait_frames, trace_frames).

        Formats:
          [8:3]               -> wait=1 beat @ bpm, trace = (240/bpm/8)*3 frames
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
            wait_frames = _seconds_to_frames(wait_sec, self.time_pre_frame)
        else:
            # no explicit wait: 1 beat at current or specified BPM
            if "#" in content and ":" not in content and content.count("#") == 1:
                # format: BPM#seconds  e.g. 160#2
                bp_s, sec_s = content.split("#", 1)
                trace_bpm = float(bp_s)
                trace_sec = float(sec_s)
                wait_frames = _seconds_to_frames(240.0 / trace_bpm, self.time_pre_frame)
                return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

            if "#" in content:
                # BPM#divider:mult
                bp_s, rest = content.split("#", 1)
                trace_bpm = float(bp_s)
                wait_frames = _seconds_to_frames(240.0 / trace_bpm, self.time_pre_frame)
                trace_sec = _length_to_seconds(rest, trace_bpm)
                return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

            # plain divider:mult  -> wait = 1 beat at bpm
            wait_frames = _seconds_to_frames(240.0 / bpm, self.time_pre_frame) if bpm > 0 else 0
            trace_sec = _length_to_seconds(content, bpm)
            return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

        # trace_rest: can be '#seconds', 'BPM#divider:mult', or 'divider:mult'
        if trace_rest.startswith("#"):
            trace_sec = float(trace_rest[1:])
            return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

        if "#" in trace_rest:
            bp_s, rest = trace_rest.split("#", 1)
            trace_bpm = float(bp_s)
            trace_sec = _length_to_seconds(rest, trace_bpm)
            return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

        trace_sec = _length_to_seconds(trace_rest, bpm)
        return wait_frames, _seconds_to_frames(trace_sec, self.time_pre_frame)

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

    def parse(self, text: str):
        self.chart = Chart(all_levels=[None] * 7)

        title_match = re.search(r"&title=([^\n]+)", text)
        if title_match:
            self.chart.title = title_match.group(1).strip()

        artist_match = re.search(r"&artist=([^\n]+)", text)
        if artist_match:
            self.chart.artist = artist_match.group(1).strip()

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
            frames_by_idx: dict[int, list[Note]] = {}
            frame_time_by_idx: dict[int, float] = {}

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

                frame_idx = int(round(self.current_time / self.time_pre_frame))
                frames_by_idx.setdefault(frame_idx, []).extend(notes)
                frame_time_by_idx.setdefault(frame_idx, self.current_time)
                self.current_time += current_per_comma_length

            frames = [
                Frame(
                    frame_idx=frame_idx,
                    notes=tuple(frames_by_idx[frame_idx]),
                    time_sec=frame_time_by_idx[frame_idx],
                )
                for frame_idx in sorted(frames_by_idx)
            ]
            level_name = f"level_{level + 1}"
            level_query_match = re.search(rf"&lv_{level}=([0-9.]+)", text)
            level_query = float(level_query_match.group(1)) if level_query_match else None
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
            prev_idx: int | None = None
            for frame in level.frames:
                for note in frame.notes:
                    self._eval_collect_note(note, extremes)
                if prev_idx is not None:
                    self._update_extremes(extremes, "frame_gap", frame.frame_idx - prev_idx)
                prev_idx = frame.frame_idx
        return extremes

    # ── tensor export ─────────────────────────────────────────────────────

    # Slot layout (13 columns):
    #   0  Δt            (int)  relative time in frames
    #   1  NoteType      (int)  NoteType enum value
    #   2  Lane          (int)  TapType enum value for TAP/HOLD/SLIDE start
    #   3  TouchType     (int)  TouchType enum value; 0 if not touch
    #   4  holdTime      (int)  HOLD / TOUCH_HOLD duration in frames; 0 otherwise
    #   5  SlideShape    (int)  SlideShape enum value; 0 if not slide
    #   6  wait_duration (int)  slide segment wait in frames; 0 otherwise
    #   7  trace_duration (int) slide segment trace in frames; 0 otherwise
    #   8  start_lane    (int)  TapType enum value; 0 if not slide
    #   9  end_lane      (int)  TapType enum value; 0 if not slide
    #   10 middle_lane   (int)  TapType enum value; 0 if not GrandV
    #   11 slide_features (int) packed bools:
    #         bit0 isClockwise-cw   (1<<0)
    #         bit1 isClockwise-ccw  (1<<1)
    #         bit2 isForceStar      (1<<2)
    #         bit3 isFakeRotate     (1<<3)
    #         bit4 isSlideBreak     (1<<4)
    #         bit5 isSlideNoHead    (1<<5)
    #   12 modifiers     (int)  packed bools:
    #         bit0 isBreak     (1<<0)
    #         bit1 isEx        (1<<1)
    #         bit2 isFirework  (1<<2)
    _SLOT_DIMS = 13

    @staticmethod
    def _pack_slide_features(seg) -> int:
        v = 0
        if seg.isClockwise is True:
            v |= 1 << 0
        elif seg.isClockwise is False:
            v |= 1 << 1
        if seg.isForceStar:
            v |= 1 << 2
        if seg.isFakeRotate:
            v |= 1 << 3
        if seg.isSlideBreak:
            v |= 1 << 4
        if seg.isSlideNoHead:
            v |= 1 << 5
        return v

    @staticmethod
    def _pack_modifiers(note) -> int:
        v = 0
        if note.isBreak:
            v |= 1 << 0
        if note.isEx:
            v |= 1 << 1
        if note.type in (NoteType.TOUCH, NoteType.TOUCH_HOLD) and note.data.isFirework:
            v |= 1 << 2
        return v

    def _note_to_slots(self, note, delta_t: int) -> list[list[int]]:
        """Convert a Note into one or more slot rows. Slides expand per segment."""
        n_type = note.type.value
        modifiers = self._pack_modifiers(note)

        if note.type == NoteType.TAP:
            return [[delta_t, n_type, note.data.value, 0, 0, 0, 0, 0, 0, 0, 0, 0, modifiers]]

        if note.type == NoteType.HOLD:
            return [[
                delta_t, n_type,
                note.data.lane.value,
                0, note.data.holdTime,
                0, 0, 0, 0, 0, 0, 0, modifiers,
            ]]

        if note.type == NoteType.TOUCH:
            return [[delta_t, n_type, 0, note.data.Touch_area.value, 0, 0, 0, 0, 0, 0, 0, 0, modifiers]]

        if note.type == NoteType.TOUCH_HOLD:
            ht = note.data.holdTime if note.data.holdTime is not None else 0
            return [[
                delta_t, n_type,
                0, note.data.Touch_area.value,
                ht, 0, 0, 0, 0, 0, 0, 0, modifiers,
            ]]

        if note.type == NoteType.SLIDE:
            rows: list[list[int]] = []
            for i, seg in enumerate(note.data):
                d = delta_t if i == 0 else 0
                rows.append([
                    d, n_type,
                    seg.start_lane.value,
                    0, 0,
                    seg.shape.value,
                    seg.wait_duration, seg.trace_duration,
                    seg.start_lane.value, seg.end_lane.value,
                    seg.middle_lane.value if seg.middle_lane is not None else 0,
                    self._pack_slide_features(seg),
                    modifiers,
                ])
            return rows

        return []

    def to_tensor(self, level_idx: int = 4) -> torch.Tensor:
        """
        Export a single level to a [N, 13] int64 tensor. See _SLOT_DIMS for layout.

        level_idx: 0..6 → lv_1..lv_7 (default 4 = lv_5 = MASTER).
        Returns an empty [0, 13] tensor if the level is missing or has no notes.
        Δt encoding:
          - First note in chart: frame index of its frame
          - First note in each subsequent frame: frame_gap from previous frame
          - Simultaneous notes (same frame, index > 0): 0
        """
        if not (0 <= level_idx < len(self.chart.all_levels)):
            raise ValueError(f"level_idx {level_idx} out of range")
        level = self.chart.all_levels[level_idx]
        if level is None:
            return torch.zeros((0, self._SLOT_DIMS), dtype=torch.int64)

        rows: list[list[int]] = []
        prev_idx: int | None = None
        for frame in level.frames:
            frame_delta = frame.frame_idx if prev_idx is None else frame.frame_idx - prev_idx
            for i_note, note in enumerate(frame.notes):
                # First note in frame carries the time gap; simultaneous
                # notes (same frame) get Δt=0 so the decoder learns they
                # are co-occurring rather than spread across frame_delta.
                delta = frame_delta if i_note == 0 else 0
                rows.extend(self._note_to_slots(note, delta))
            prev_idx = frame.frame_idx

        if not rows:
            return torch.zeros((0, self._SLOT_DIMS), dtype=torch.int64)
        return torch.tensor(rows, dtype=torch.int64)

    # ── tensor import ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_slide_features(v: int) -> dict:
        return {
            "isClockwise": True if v & (1 << 0) else (False if v & (1 << 1) else None),
            "isForceStar": bool(v & (1 << 2)),
            "isFakeRotate": bool(v & (1 << 3)),
            "isSlideBreak": bool(v & (1 << 4)),
            "isSlideNoHead": bool(v & (1 << 5)),
        }

    @staticmethod
    def _unpack_modifiers(v: int) -> dict:
        return {
            "isBreak": bool(v & (1 << 0)),
            "isEx": bool(v & (1 << 1)),
            "isFirework": bool(v & (1 << 2)),
        }

    def parse_from_tensor(self, tensor: torch.Tensor, level_idx: int = 4, title: str = "generated", artist: str | None = None, level_query: float | None = None) -> "compiler":
        """
        Import a [N, 13] int64 tensor into the compiler's chart at level_idx.
        Inverse of to_tensor(). Returns self for chaining: parser.parse_from_tensor(t).generate()
        
        If chart already exists, preserves title/artist unless overridden.
        """
        if tensor.ndim != 2 or tensor.shape[1] != self._SLOT_DIMS:
            raise ValueError(f"Expected tensor shape [N, {self._SLOT_DIMS}], got {tensor.shape}")

        if not (0 <= level_idx < 7):
            raise ValueError(f"level_idx {level_idx} out of range 0..6")

        rows = tensor.tolist()
        frames_by_idx: dict[int, list[Note]] = {}
        frame_time_by_idx: dict[int, float] = {}

        current_frame_idx = 0
        i = 0
        n = len(rows)

        while i < n:
            row = rows[i]
            dt, note_type_val = row[0], row[1]
            current_frame_idx += dt
            note_type = NoteType(note_type_val)
            modifiers = self._unpack_modifiers(row[12])
            is_firework = modifiers.pop("isFirework", False)

            if note_type == NoteType.TAP:
                lane = TapType(row[2])
                note = Note(note_type, lane, **modifiers)
                frames_by_idx.setdefault(current_frame_idx, []).append(note)
                frame_time_by_idx.setdefault(current_frame_idx, current_frame_idx * self.time_pre_frame)
                i += 1

            elif note_type == NoteType.HOLD:
                lane = TapType(row[2])
                hold_time = row[4]
                note = Note(note_type, Hold_data(lane, hold_time), **modifiers)
                frames_by_idx.setdefault(current_frame_idx, []).append(note)
                frame_time_by_idx.setdefault(current_frame_idx, current_frame_idx * self.time_pre_frame)
                i += 1

            elif note_type == NoteType.TOUCH:
                touch_area = TouchType(row[3])
                note = Note(note_type, Touch_data(touch_area, isFirework=is_firework), **modifiers)
                frames_by_idx.setdefault(current_frame_idx, []).append(note)
                frame_time_by_idx.setdefault(current_frame_idx, current_frame_idx * self.time_pre_frame)
                i += 1

            elif note_type == NoteType.TOUCH_HOLD:
                touch_area = TouchType(row[3])
                hold_time = row[4] if row[4] > 0 else None
                note = Note(note_type, Touch_data(touch_area, isFirework=is_firework, holdTime=hold_time), **modifiers)
                frames_by_idx.setdefault(current_frame_idx, []).append(note)
                frame_time_by_idx.setdefault(current_frame_idx, current_frame_idx * self.time_pre_frame)
                i += 1

            elif note_type == NoteType.SLIDE:
                r = rows[i]
                sf = self._unpack_slide_features(r[11])
                shape = SlideShape(r[5])
                start_lane = TapType(r[8])
                end_lane = TapType(r[9])
                middle_lane = TapType(r[10]) if r[10] != 0 else None
                seg = SlideSegment(
                    shape=shape,
                    start_lane=start_lane,
                    end_lane=end_lane,
                    wait_duration=r[6],
                    trace_duration=r[7],
                    isClockwise=sf["isClockwise"],
                    middle_lane=middle_lane,
                    isForceStar=sf["isForceStar"],
                    isFakeRotate=sf["isFakeRotate"],
                    isSlideBreak=sf["isSlideBreak"],
                    isSlideNoHead=sf["isSlideNoHead"],
                )
                note = Note(note_type, [seg], **modifiers)
                frames_by_idx.setdefault(current_frame_idx, []).append(note)
                frame_time_by_idx.setdefault(current_frame_idx, current_frame_idx * self.time_pre_frame)
                i += 1

            else:
                i += 1

        frames = [
            Frame(
                frame_idx=idx,
                notes=tuple(frames_by_idx[idx]),
                time_sec=frame_time_by_idx[idx],
            )
            for idx in sorted(frames_by_idx)
        ]

        # Preserve existing chart metadata if available
        if self.chart is not None:
            chart_title = self.chart.title if title == "generated" else title
            chart_artist = self.chart.artist if artist is None else artist
        else:
            chart_title = title
            chart_artist = artist or "default"

        if self.chart is None or self.chart.all_levels == []:
            self.chart = Chart(all_levels=[None] * 7, title=chart_title, artist=chart_artist)
        elif len(self.chart.all_levels) <= level_idx:
            self.chart.all_levels.extend([None] * (level_idx + 1 - len(self.chart.all_levels)))

        self.chart.all_levels[level_idx] = Level(
            level_name=f"level_{level_idx + 1}",
            level_query=level_query if level_query is not None else 0.0,
            frames=frames,
        )
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
    def _duration_to_notation(seconds: float, bpm: float) -> str:
        """Convert duration in seconds to simai [X:Y] notation at given BPM.
        Tries power-of-2 denominators, falls back to #seconds."""
        if seconds <= 0:
            return "1:1"
        beats = seconds * bpm / 240.0
        # tolerance: frame quantization can cause ~0.3 beats of error at 12000 BPM
        for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]:
            y = beats * x
            yi = int(round(y))
            if yi > 0 and abs(y - yi) < 0.5:
                return f"{x}:{yi}"
        # fallback: #seconds
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
            hold_frames = note.data.holdTime
            if hold_frames and hold_frames > 0:
                hold_sec = hold_frames * self.time_pre_frame
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
            hold_frames = note.data.holdTime
            if hold_frames is not None and hold_frames > 0:
                hold_sec = hold_frames * self.time_pre_frame
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
                trace_sec = seg.trace_duration * self.time_pre_frame
                wait_sec = seg.wait_duration * self.time_pre_frame
                default_wait = 240.0 / bpm  # 1 beat at generation BPM
                trace_str = self._duration_to_notation(trace_sec, bpm)
                # strip leading '#' for use inside [wait##trace]
                trace_val = trace_str.lstrip('#')
                if abs(wait_sec - default_wait) < 0.005:
                    parts.append(f"[{trace_str}]")
                else:
                    ws = f"{wait_sec:.6f}"
                    ws = ws.rstrip('0').rstrip('.')
                    parts.append(f"[{ws}##{trace_val}]")

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
        per_comma_frames = (240.0 / gen_bpm / gen_divider) / self.time_pre_frame

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

            for frame in sorted(level.frames, key=lambda f: f.frame_idx):
                if not frame.notes:
                    continue

                # reconstruct note text from parsed Note objects
                combined = "/".join(self._note_to_text(n, gen_bpm) for n in frame.notes)

                # derive comma count from frame_idx via per_comma_frames
                target_commas = max(0, int(round(frame.frame_idx / per_comma_frames)))
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

def main():
    import difflib

    charts_dir = Path(__file__).resolve().parent.parent / "charts"
    tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
    found = errors = 0
    diffs = 0
    overall: dict[str, dict[str, int]] = {}
    for chart_dir, _dirs, files in charts_dir.walk():
        if "maidata.txt" not in files:
            continue
        rel = chart_dir.relative_to(charts_dir)
        out_path = tmp_dir / rel / "maidata.txt"
        try:
            text = (chart_dir / "maidata.txt").read_text(encoding="utf-8")
            parser = compiler(hop_length=512, sample_rate=44100)
            extremes = parser.eval(text)
            out_path.parent.mkdir(parents=True, exist_ok=True)
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

            # ── parse vs parse_from_tensor 全 level 对比 ──
            title = parser.chart.title
            artist = parser.chart.artist
            text_a = parser.generate()
            tensors: dict[int, tuple[torch.Tensor, float]] = {}
            for li in range(7):
                lv = parser.chart.all_levels[li]
                if lv is not None:
                    t = parser.to_tensor(level_idx=li)
                    tensors[li] = (t, lv.level_query)
            parser.chart = None
            for li, (t, lq) in tensors.items():
                parser.parse_from_tensor(t, level_idx=li, title=title, artist=artist, level_query=lq)
            text_b = parser.generate()
            out_path.write_text(text_b, encoding="utf-8")

            # 过滤仅逗号数量不同的行
            a_lines = text_a.splitlines()
            b_lines = text_b.splitlines()
            max_len = max(len(a_lines), len(b_lines))
            a_lines += [''] * (max_len - len(a_lines))
            b_lines += [''] * (max_len - len(b_lines))
            filtered_a: list[str] = []
            filtered_b: list[str] = []
            has_real_diff = False
            for la, lb in zip(a_lines, b_lines):
                if la == lb:
                    filtered_a.append(la)
                    filtered_b.append(lb)
                elif la.rstrip(',') == lb.rstrip(','):
                    filtered_a.append(la)
                    filtered_b.append(lb)
                else:
                    has_real_diff = True
                    filtered_a.append(la)
                    filtered_b.append(lb)

            if has_real_diff:
                diffs += 1
                print(f"\n[DIFF] {rel}")
                diff = difflib.unified_diff(
                    [l + '\n' for l in filtered_a],
                    [l + '\n' for l in filtered_b],
                    fromfile="parse → generate",
                    tofile="parse → tensor → parse_from_tensor → generate",
                )
                for line in diff:
                    print("  " + line.rstrip())

        except Exception as e:
            import traceback
            errors += 1
            print(f"[ERR] {rel}: {e}")
            traceback.print_exc()

    print(f"\nDone: {found} parsed, {errors} errors, {diffs} diffs")
    print(f"Overall extremes per field: {overall}")


if __name__ == "__main__":
    main()
