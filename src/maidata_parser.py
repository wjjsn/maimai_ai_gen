import math
import re
from chart import Chart, Frame, HoldData, Level, Note, NoteType, SlideSegment, SlideShape, TapType, TouchData, TouchType

__all__ = ["parse_maidata", "generate_maidata"]
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

class _Parser:
    def __init__(self):
        self.chart = Chart(all_levels=[None] * 7)
        self.current_time = 0.0  # seconds
        self._warned: set[str] = set()

    def _log_once(self, kind: str, message: str, *, warning: bool = False) -> None:
        if kind in self._warned:
            return
        self._warned.add(kind)
        # 批量索引由调用方汇总这些可恢复兼容处理，避免逐首刷屏。

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
        if token[0] not in _LANE_MAP:
            raise ValueError(f"未知音符起始字符: {token}")
        return self._parse_tap(token)

    @staticmethod
    def _is_multi_tap(token: str) -> bool:
        """Check if token is a compact EACH of pure TAPs (e.g. '17' = TAP1 + TAP7)."""
        return len(re.findall(r"[1-8]", token)) >= 2 and re.fullmatch(r"(?:[1-8][bx$]*)+", token) is not None

    def _parse_note(self, token: str, bpm: float) -> list[Note] | None:
        """Parse note token, normalizing pseudo-EACH (`` ` ``) to ordinary EACH."""
        if not token or token == "E":
            return None
        # 反引号是伪 EACH 分隔符，和普通 EACH 一样拆分，不需要额外日志。
        parts = re.split(r"[/`]", token)
        notes: list[Note] = []
        for part in parts:
            part = part.strip()
            if not part:
                self._log_once("empty-each", f"已忽略 EACH 空分支，示例: {token}", warning=True)
                continue
            # compact EACH: '17' -> TAP at 1 + TAP at 7
            if self._is_multi_tap(part):
                notes.extend(self._parse_tap(atom) for atom in re.findall(r"[1-8][bx$]*", part))
            else:
                n = self._parse_single_note(part, bpm)
                if n is not None:
                    notes.append(n)
        return notes if notes else None

    # ── individual note type parsers ──────────────────────────────────────

    def _parse_tap(self, token: str) -> Note:
        if token[0] not in _LANE_MAP:
            raise ValueError(f"TAP 轨道无效: {token}")
        bracket = re.search(r"\[[^]]+\]$", token)
        if bracket:
            self._log_once("tap-duration", f"已忽略 TAP 后的孤立时长，示例: {token}", warning=True)
            token = token[:bracket.start()]
        if any(ch not in "bx$" for ch in token[1:]):
            raise ValueError(f"TAP 含未知修饰符: {token}")
        # 星形标记不改变 TAP 的内部表达，直接归一化。
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
                data=HoldData(lane=lane, holdTime=hold_sec),
                isBreak=is_break,
                isEx=is_ex,
            )
        else:
            # 无时长的伪 HOLD 按 TAP 归一化。
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
        if area_str not in _TOUCH_MAP:
            raise ValueError(f"TOUCH 区域无效: {token}")

        if has_h:
            m = re.search(r"\[([^\]]+)\]", token)
            if m:
                # TOUCH_HOLD with bracket
                hold_sec = _length_to_seconds(m.group(1), bpm)
                return Note(
                    type=NoteType.TOUCH_HOLD,
                    data=TouchData(Touch_area=area, isFirework=has_f, holdTime=hold_sec),
                )
            else:
                # 无时长的伪 TOUCH HOLD 按 TOUCH 归一化。
                return Note(
                    type=NoteType.TOUCH,
                    data=TouchData(Touch_area=area, isFirework=has_f),
                )
        else:
            return Note(
                type=NoteType.TOUCH,
                data=TouchData(Touch_area=area, isFirework=has_f),
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
        if not segments:
            raise ValueError(f"无法解析 Slide: {token}")
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
                is_slide_break = True
                i += 1
            elif t[i] in ("?", "!"):
                is_no_head = True
                i += 1

        # '*' separator between multiple slides at same start
        # we handle it in the outer loop

        cur_start = start_lane
        previous_shape: SlideShape | None = None
        previous_cw: bool | None = None

        while i < n:
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
                    is_slide_break = True
                    i += 1
                else:
                    is_no_head = True
                    i += 1

            # skip '*' separator (multiple slide)
            if t[i] == "*":
                i += 1
                if i >= n:
                    raise ValueError(f"Slide 末尾存在空分支: {t}")
                # reset cur_start to original start for multiple slides
                # (multiple slides share the same start point)
                # actually after *, next segment starts from the *original* start
                cur_start = start_lane

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
                        is_slide_break = True
                        i += 1
                    else:
                        is_no_head = True
                        i += 1

            # try to match a shape
            shape = self._match_shape(t, i)
            if shape is None:
                if previous_shape is not None and t[i] in "12345678":
                    shape = previous_shape
                    is_cw = previous_cw
                    self._log_once(
                        "missing-slide-shape",
                        f"链式 Slide 缺少形状符，已复用上一段形状，示例: {t}",
                        warning=True,
                    )
                else:
                    raise ValueError(f"Slide 含无法识别的片段: {t[i:]} ({t})")
            else:
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

            # 部分谱面把 Break/EX 写在形状与终点之间，如 2pbx7b[8:3]。
            while i < n and t[i] in ("b", "x"):
                is_slide_break |= t[i] == "b"
                is_ex |= t[i] == "x"
                i += 1

            # For GrandV: middle_lane is an extra digit before end lane
            middle_lane = None
            if shape == SlideShape.GrandV:
                if i < n and t[i] in "12345678":
                    middle_lane = _lane(t[i])
                    i += 1

            # end lane
            if i >= n or t[i] not in "12345678":
                raise ValueError(f"Slide 缺少终点: {t[i:]} ({t})")
            end_lane = _lane(t[i])
            i += 1

            # optional 'b' break marker before bracket
            while i < n and t[i] in ("b", "x"):
                is_slide_break |= t[i] == "b"
                is_ex |= t[i] == "x"
                i += 1

            # optional [wait##trace] for this segment
            wait_sec = 240.0 / bpm if bpm > 0 else 0.0
            trace_sec = 0.0
            is_default_wait = True
            if i < n and t[i] == "[":
                j = t.index("]", i)
                bracket_content = t[i + 1 : j]
                wait_sec, trace_sec = self._parse_slide_bracket(bracket_content, bpm)
                # 默认等待是当前 BPM 的一拍；指定其他 BPM 也会改变等待时长。
                is_default_wait = abs(wait_sec - 240.0 / bpm) < 1e-6 if bpm > 0 else wait_sec == 0.0
                i = j + 1

            while i < n and t[i] in ("b", "x"):
                is_slide_break |= t[i] == "b"
                is_ex |= t[i] == "x"
                i += 1

            seg = SlideSegment(
                shape=shape,
                start_lane=cur_start,
                end_lane=end_lane,
                wait_duration=wait_sec,
                trace_duration=trace_sec,
                is_default_wait=is_default_wait,
                isClockwise=is_cw,
                middle_lane=middle_lane,
                isForceStar=is_force_star,
                isFakeRotate=is_fake_rotate,
                isSlideBreak=is_slide_break,
                isSlideNoHead=is_no_star or is_no_head,
            )
            segments.append(seg)
            cur_start = end_lane  # chaining: next starts where this ended
            previous_shape = shape
            previous_cw = is_cw

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

    def parse(self, text: str) -> Chart:
        self.chart = Chart(all_levels=[None] * 7)
        self._warned.clear()

        title_match = re.search(r"&title=([^\n]+)", text)
        if title_match:
            self.chart.title = title_match.group(1).strip()

        artist_match = re.search(r"&artist=([^\n]+)", text)
        if artist_match:
            self.chart.artist = artist_match.group(1).strip()

        first_match = re.search(r"(?m)^&first=([^\r\n]+)", text)
        if first_match:
            try:
                self.chart.first_sec = float(first_match.group(1).strip())
            except ValueError as error:
                raise ValueError(f"谱面 &first 无效: {first_match.group(1)!r}") from error
            if not math.isfinite(self.chart.first_sec):
                raise ValueError("谱面 &first 必须是有限数字")

        for level in range(0, 7):
            level_match = re.search(
                rf"&inote_{level}=([\s\S]*?)(?=^&|\Z)",
                text,
                re.MULTILINE,
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

            for slot_idx, raw_note in enumerate(chart_content_strip):
                # strip comments (|| to end of line)
                if "||" in raw_note:
                    raw_note = raw_note[:raw_note.index("||")]
                raw_note = re.sub(r"\s+", "", raw_note)
                if not raw_note:
                    self.current_time += current_per_comma_length
                    continue
                if raw_note == "E":
                    break

                try:
                    current_bpm, current_length_divider, current_per_comma_length = (
                        self._parse_current_time_per_comma(
                            raw_note,
                            current_bpm,
                            current_length_divider,
                            current_per_comma_length,
                        )
                    )
                except (ValueError, ZeroDivisionError) as error:
                    raise ValueError(
                        f"歌曲={self.chart.title!r} 难度={level} 槽位={slot_idx} "
                        f"时间={self.current_time:.6f}s 时值定义无效: {raw_note!r}"
                    ) from error

                cleaned = re.sub(r"\([^)]*\)", "", raw_note)
                cleaned = re.sub(r"\{[^}]*\}", "", cleaned).strip()
                # 容忍时值定义后多写的右花括号，例如 ``{2}}8h[...]``。
                stripped = cleaned.lstrip("}")
                if stripped != cleaned:
                    self._log_once("extra-brace", f"已忽略多余右花括号，示例: {raw_note}", warning=True)
                    cleaned = stripped
                if not cleaned or cleaned == "E":
                    self.current_time += current_per_comma_length
                    continue

                try:
                    notes = self._parse_note(cleaned, current_bpm)
                except (KeyError, ValueError, IndexError) as error:
                    raise ValueError(
                        f"歌曲={self.chart.title!r} 难度={level} 槽位={slot_idx} "
                        f"时间={self.current_time:.6f}s 音符解析失败: {cleaned!r}"
                    ) from error
                if notes is None:
                    raise ValueError(
                        f"歌曲={self.chart.title!r} 难度={level} 槽位={slot_idx} "
                        f"时间={self.current_time:.6f}s 未解析出音符: {cleaned!r}"
                    )

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
            self.chart.all_levels[level] = Level(
                level_name=level_name, level_query=level_query, frames=frames
            )

        return self.chart

    '''旧 token、tensor 和训练辅助逻辑已移除。
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

    '''
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
        dividers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]
        for x in dividers:
            y = beats * x
            yi = int(round(y))
            if yi > 0 and abs(y - yi) < 1e-6:
                return f"{x}:{yi}"
        for x in dividers:
            y = beats * x
            yi = int(round(y))
            if yi > 0 and abs(y - yi) / yi < 0.02:
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
                if wait_sec <= 0:
                    ts = f"{trace_sec:.6f}".rstrip('0').rstrip('.')
                    parts.append(f"[0###{ts}]")
                elif abs(wait_sec - default_wait) < 0.005:
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
        lines.append(f"&first={self.chart.first_sec:g}")

        gen_bpm = 12000.0
        gen_divider = 4.0
        per_comma_time = 240.0 / gen_bpm / gen_divider  # 0.005 seconds

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
                    prefix = "(12000){4}" + "," * gap_commas
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


def parse_maidata(text: str) -> Chart:
    return _Parser().parse(text)


def generate_maidata(chart: Chart) -> str:
    parser = _Parser()
    parser.chart = chart
    return parser.generate()


def _self_check() -> None:
    text = "&title=test\n&artist=tester\n&first=0.125\n&lv_5=14\n&inote_5=(12000){4},1,2h[4:1],E"
    chart = parse_maidata(text)
    assert chart.title == "test" and chart.first_sec == 0.125
    assert chart.all_levels[5] is not None
    generated = generate_maidata(chart)
    restored = parse_maidata(generated)
    assert restored.first_sec == 0.125 and len(restored.all_levels[5].frames) == 2
    print("[maidata-parser] 自检通过")


if __name__ == "__main__":
    _self_check()


'''旧批处理和兼容性自检已移除。
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

    chart = compiler().parse(
        "&title=test\n&lv_5=14\n&inote_5=(120){4}1`2`3/4,5h/Chf,2pbx7b[8:3]*p8b[8:3],1?-5[8:1],E"
    )
    frames = chart.all_levels[5].frames
    assert [note.data for note in frames[0].notes] == [
        TapType.LANE1, TapType.LANE2, TapType.LANE3, TapType.LANE4,
    ]
    assert frames[1].notes[0] == Note(NoteType.TAP, TapType.LANE5)
    assert frames[1].notes[1] == Note(NoteType.TOUCH, Touch_data(TouchType.C, isFirework=True))
    slide = frames[2].notes[0]
    assert slide.type is NoteType.SLIDE and slide.isBreak and slide.isEx
    assert len(slide.data) == 2
    assert [(seg.start_lane, seg.end_lane) for seg in slide.data] == [
        (TapType.LANE2, TapType.LANE7), (TapType.LANE2, TapType.LANE8),
    ]
    assert all(seg.isSlideBreak for seg in slide.data)
    assert frames[3].notes[0].data[0].isSlideNoHead
    assert compiler().parse(
        "&title=test\n&lv_5=14\n&inote_5=(120){2}}8h[64:47],E"
    ).all_levels[5].frames[0].notes[0].type is NoteType.HOLD
    try:
        compiler()._parse_slide("2p", 120)
    except ValueError:
        pass
    else:
        raise AssertionError("空 Slide 必须拒绝")
    print("[maidata-parser] 自检通过")


'''
