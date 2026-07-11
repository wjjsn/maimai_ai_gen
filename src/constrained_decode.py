"""maimai token 语法约束解码。

当前硬约束：
- 序列只能以 SOS 开始；SOS 后只能开始新帧或直接 EOS。
- FRAME_START 后只能输出时间戳 TS。
- 帧时间戳相对当前 segment 不能倒退；允许相等，因为训练集中存在同一时间点拆成多帧。
- 帧内只能输出 NOTE_TAP / NOTE_TOUCH / NOTE_HOLD / NOTE_SLIDE 或 FRAME_END。
- 每帧最多允许 33 个 note；这是当前训练集实测最大值，不能设成 2。
- 每帧 TAP/HOLD/SLIDE 合计最多允许 4 个；这是当前训练集实测最大值，TOUCH 不计入此限制。
- TAP: NOTE_TAP -> LANE -> 可选 IS_BREAK/IS_EX -> 下一个 note 或 FRAME_END。
- HOLD: NOTE_HOLD -> LANE -> TS(duration) -> 可选 IS_BREAK/IS_EX -> 下一个 note 或 FRAME_END。
- TOUCH: NOTE_TOUCH -> TOUCH_AREA -> 可选 TS(duration) -> 可选 IS_FIREWORK -> 下一个 note 或 FRAME_END。
- SLIDE: NOTE_SLIDE -> 可选 IS_BREAK/IS_EX -> 可选空 slide 直接 FRAME_END，或一个/多个 slide segment。
- Slide segment: SEGMENT_START -> SHAPE -> start LANE -> end LANE -> GrandV 必须 middle LANE -> TS(wait) -> TS(trace) -> 可选段属性 -> SEGMENT_END。
- 段属性里 IS_CW/IS_CCW 互斥，同类属性不能重复。
- FRAME_END 后只能开始下一帧或 EOS。

这个约束器只保证 token 结构合法，不保证音乐内容合理，也不保证生成结果和训练集一致。
"""

from maidata_parser import (
    EOS,
    FRAME_END,
    FRAME_START,
    IS_BREAK,
    IS_CCW,
    IS_CW,
    IS_EX,
    IS_FAKE_ROTATE,
    IS_FIREWORK,
    IS_FORCE_STAR,
    IS_SLIDE_BREAK,
    IS_SLIDE_NO_HEAD,
    LANE_BASE,
    NOTE_HOLD,
    NOTE_SLIDE,
    NOTE_TAP,
    NOTE_TOUCH,
    PAD,
    SEGMENT_END,
    SEGMENT_START,
    SLIDE_SHAPE_BASE,
    SOS,
    TOUCH_BASE,
    TS_BASE,
)


LANES = tuple(range(LANE_BASE, LANE_BASE + 8))
TOUCHES = tuple(range(TOUCH_BASE, TOUCH_BASE + 33))
SHAPES = tuple(range(SLIDE_SHAPE_BASE, SLIDE_SHAPE_BASE + 11))
GRAND_V = SLIDE_SHAPE_BASE + 3
TIMES = tuple(range(TS_BASE, TS_BASE + 3000))
NOTE_TYPES = (NOTE_TAP, NOTE_TOUCH, NOTE_HOLD, NOTE_SLIDE)
TAP_ATTRS = (IS_BREAK, IS_EX)
SLIDE_NOTE_ATTRS = (IS_BREAK, IS_EX)
SLIDE_SEG_ATTRS = (IS_CW, IS_CCW, IS_FORCE_STAR, IS_FAKE_ROTATE, IS_SLIDE_BREAK, IS_SLIDE_NO_HEAD)


def allowed_tokens(
    tokens: list[int],
    max_notes_per_frame: int = 33,
    max_action_notes_per_frame: int = 4,
    min_frame_time: int = 0,
    max_frame_time: int = 2999,
) -> tuple[int, ...]:
    """返回下一步允许的 token。只做语法约束，不判断音乐合理性。"""
    if not tokens:
        return (SOS,)
    if tokens[-1] in (EOS, PAD):
        return ()
    if tokens == [SOS]:
        return (FRAME_START, EOS)

    start = _last_frame_start(tokens)
    if start < 0:
        return (FRAME_START, EOS)

    frame = tokens[start:]
    if len(frame) == 1:
        return _times_from(max(_previous_frame_time(tokens[:start]), min_frame_time), max_frame_time)
    if len(frame) == 2:
        return NOTE_TYPES

    return _allowed_in_frame(frame[2:], max_notes_per_frame, max_action_notes_per_frame)


def _last_frame_start(tokens: list[int]) -> int:
    try:
        end = len(tokens) - 1 - tokens[::-1].index(FRAME_END)
    except ValueError:
        end = 0
    try:
        return len(tokens) - 1 - tokens[::-1].index(FRAME_START)
    except ValueError:
        return -1 if end == 0 else end + 1


def _previous_frame_time(tokens: list[int]) -> int:
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] == FRAME_START and TS_BASE <= tokens[i + 1] < TS_BASE + 3000:
            return tokens[i + 1] - TS_BASE
    return 0


def _times_from(min_time: int, max_time: int = 2999) -> tuple[int, ...]:
    return tuple(range(TS_BASE + min_time, TS_BASE + min(max_time, 2999) + 1))


def _allowed_in_frame(body: list[int], max_notes_per_frame: int, max_action_notes_per_frame: int) -> tuple[int, ...]:
    if not body:
        return NOTE_TYPES
    i = 0
    notes = 0
    action_notes = 0
    while i < len(body):
        tok = body[i]
        rest = body[i:]
        if tok == NOTE_TAP:
            done, allowed = _consume_tap(rest)
        elif tok == NOTE_HOLD:
            done, allowed = _consume_hold(rest)
        elif tok == NOTE_TOUCH:
            done, allowed = _consume_touch(rest)
        elif tok == NOTE_SLIDE:
            done, allowed = _consume_slide(rest)
        elif tok == FRAME_END:
            return (FRAME_START, EOS)
        else:
            return (FRAME_END,)

        if allowed:
            if notes >= max_notes_per_frame:
                return (FRAME_END,)
            if action_notes >= max_action_notes_per_frame:
                return tuple(t for t in allowed if t not in (NOTE_TAP, NOTE_HOLD, NOTE_SLIDE)) or (FRAME_END,)
            return allowed
        i += done
        notes += 1
        if tok in (NOTE_TAP, NOTE_HOLD, NOTE_SLIDE):
            action_notes += 1

    if notes >= max_notes_per_frame:
        return (FRAME_END,)
    if action_notes >= max_action_notes_per_frame:
        return (NOTE_TOUCH, FRAME_END)
    return NOTE_TYPES + (FRAME_END,)


def _consume_tap(tokens: list[int]) -> tuple[int, tuple[int, ...]]:
    if len(tokens) == 1:
        return 0, LANES
    if tokens[1] not in LANES:
        return 0, LANES
    i = 2
    used = set()
    while i < len(tokens) and tokens[i] in TAP_ATTRS and tokens[i] not in used:
        used.add(tokens[i])
        i += 1
    if i < len(tokens):
        return i, ()
    return i, tuple(t for t in TAP_ATTRS if t not in used) + NOTE_TYPES + (FRAME_END,)


def _consume_hold(tokens: list[int]) -> tuple[int, tuple[int, ...]]:
    if len(tokens) == 1:
        return 0, LANES
    if tokens[1] not in LANES:
        return 0, LANES
    if len(tokens) == 2:
        return 0, TIMES
    if tokens[2] not in TIMES:
        return 0, TIMES
    i = 3
    used = set()
    while i < len(tokens) and tokens[i] in TAP_ATTRS and tokens[i] not in used:
        used.add(tokens[i])
        i += 1
    if i < len(tokens):
        return i, ()
    return i, tuple(t for t in TAP_ATTRS if t not in used) + NOTE_TYPES + (FRAME_END,)


def _consume_touch(tokens: list[int]) -> tuple[int, tuple[int, ...]]:
    if len(tokens) == 1:
        return 0, TOUCHES
    if tokens[1] not in TOUCHES:
        return 0, TOUCHES
    i = 2
    has_time = False
    has_firework = False
    if i < len(tokens) and tokens[i] in TIMES:
        has_time = True
        i += 1
    if i < len(tokens) and tokens[i] == IS_FIREWORK:
        has_firework = True
        i += 1
    if i < len(tokens):
        return i, ()
    allowed = []
    if not has_time:
        allowed.extend(TIMES)
    if not has_firework:
        allowed.append(IS_FIREWORK)
    allowed.extend(NOTE_TYPES)
    allowed.append(FRAME_END)
    return i, tuple(allowed)


def _consume_slide(tokens: list[int]) -> tuple[int, tuple[int, ...]]:
    i = 1
    used_note_attrs = set()
    while i < len(tokens) and tokens[i] in SLIDE_NOTE_ATTRS and tokens[i] not in used_note_attrs:
        used_note_attrs.add(tokens[i])
        i += 1
    if i == len(tokens):
        return 0, tuple(t for t in SLIDE_NOTE_ATTRS if t not in used_note_attrs) + (SEGMENT_START, FRAME_END)

    saw_segment = False
    while i < len(tokens):
        if tokens[i] != SEGMENT_START:
            if tokens[i] == FRAME_END and not saw_segment:
                return i, ()
            if saw_segment:
                return i, ()
            return i, (SEGMENT_START,)
        done, allowed = _consume_slide_segment(tokens[i:])
        if allowed:
            return 0, allowed
        i += done
        saw_segment = True

    return i, (SEGMENT_START,) + NOTE_TYPES + (FRAME_END,)


def _consume_slide_segment(tokens: list[int]) -> tuple[int, tuple[int, ...]]:
    if len(tokens) == 1:
        return 0, SHAPES
    if tokens[1] not in SHAPES:
        return 0, SHAPES
    if len(tokens) == 2:
        return 0, LANES
    if tokens[2] not in LANES:
        return 0, LANES
    if len(tokens) == 3:
        return 0, LANES
    if tokens[3] not in LANES:
        return 0, LANES

    i = 4
    if tokens[1] == GRAND_V:
        if i == len(tokens):
            return 0, LANES
        if tokens[i] not in LANES:
            return 0, LANES
        i += 1
    if i == len(tokens):
        return 0, TIMES
    if tokens[i] not in TIMES:
        return 0, TIMES
    i += 1
    if i == len(tokens):
        return 0, TIMES
    if tokens[i] not in TIMES:
        return 0, TIMES
    i += 1

    used = set()
    while i < len(tokens) and tokens[i] in SLIDE_SEG_ATTRS and tokens[i] not in used:
        if tokens[i] == IS_CW and IS_CCW in used:
            break
        if tokens[i] == IS_CCW and IS_CW in used:
            break
        used.add(tokens[i])
        i += 1
    if i < len(tokens):
        if tokens[i] == SEGMENT_END:
            return i + 1, ()
        return i, ()
    allowed = tuple(t for t in SLIDE_SEG_ATTRS if t not in used) + (SEGMENT_END,)
    return i, allowed
