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
        starts_new_branch = True

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
                starts_new_branch = True

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
                starts_new_branch=starts_new_branch,
            )
            segments.append(seg)
            starts_new_branch = False
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

        if ":" in trace_rest:
            return wait_sec, _length_to_seconds(trace_rest, bpm)
        return wait_sec, float(trace_rest)

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
            elif seg.starts_new_branch:
                parts.append("*")
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
                    parts.append(f"[0##{ts}]")
                elif abs(wait_sec - default_wait) < 0.005:
                    # wait = 1 beat at gen BPM, use simple [D:M] notation
                    trace_str = self._duration_to_notation(trace_sec, bpm)
                    parts.append(f"[{trace_str}]")
                else:
                    ws = f"{wait_sec:.6f}".rstrip('0').rstrip('.')
                    ts = f"{trace_sec:.6f}".rstrip('0').rstrip('.')
                    parts.append(f"[{ws}##{ts}]")

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
    slides = parse_maidata(
        "&title=slides\n&lv_5=14\n&inote_5=(120){4}"
        "1-3[0.3##0.15],2-4[8:1],4pp4[4:1]*qq4[4:1],"
        "1-3[8:1]-1[8:1]*-5[8:1],E"
    ).all_levels[5].frames
    absolute, beats, self_loop, returned = (frame.notes[0] for frame in slides)
    assert absolute.data[0].wait_duration == 0.3
    assert absolute.data[0].trace_duration == 0.15
    assert beats.data[0].wait_duration == 2.0
    assert beats.data[0].trace_duration == 0.25
    assert [segment.starts_new_branch for segment in self_loop.data] == [True, True]
    assert [segment.starts_new_branch for segment in returned.data] == [True, False, True]
    regenerated = generate_maidata(parse_maidata(
        "&title=roundtrip\n&lv_5=14\n&inote_5=(120){4}1-3[0.3##0.15],"
        "4pp4[4:1]*qq4[4:1],1-3[8:1]-1[8:1]*-5[8:1],E"
    ))
    assert "[0.3##0.15]" in regenerated
    assert "4pp4[" in regenerated and "*qq4[" in regenerated
    assert "1-3[" in regenerated and "-1[" in regenerated and "*-5[" in regenerated
    print("[maidata-parser] 自检通过")


if __name__ == "__main__":
    import shutil

    from config import CONFIG, ROOT_DIR


    OUTPUT_DIR = ROOT_DIR / "tmp" / "maidata-12000"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        converted = 0
        errors: list[str] = []
        for source in sorted(CONFIG.paths.charts_dir.rglob("maidata.txt")):
            relative = source.relative_to(CONFIG.paths.charts_dir)
            try:
                chart = parse_maidata(source.read_text(encoding="utf-8"))
                text = generate_maidata(chart)
                parse_maidata(text)
                destination = OUTPUT_DIR / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
                converted += 1
            except Exception as error:
                errors.append(f"{relative}: {error}")
        print(f"[full-check] 转换成功={converted} 转换失败={len(errors)} 输出={OUTPUT_DIR}")
        for error in errors[:20]:
            print(f"  {error}")
        if errors:
            raise RuntimeError(f"全量谱面检查失败 {len(errors)} 首")
