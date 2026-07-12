"""maimai 谱面事件与模型 token 之间的转换。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maidata_parser import Frame, Note


PAD = 0
SOS = 1
EOS = 2
FRAME_START = 3
FRAME_END = 4
TS_BASE = 5
TS_COUNT = 3000
LANE_BASE = 3005
LANE_COUNT = 8
TOUCH_BASE = 3013
TOUCH_COUNT = 33
NOTE_TAP = 3046
NOTE_TOUCH = 3047
NOTE_HOLD = 3048
NOTE_SLIDE = 3049
SEGMENT_START = 3050
SEGMENT_END = 3051
SLIDE_SHAPE_BASE = 3052
SLIDE_SHAPE_COUNT = 11
IS_BREAK = 3063
IS_EX = 3064
IS_FIREWORK = 3065
IS_CW = 3066
IS_CCW = 3067
IS_FORCE_STAR = 3068
IS_FAKE_ROTATE = 3069
IS_SLIDE_BREAK = 3070
IS_SLIDE_NO_HEAD = 3071
VOCAB_SIZE = 3072

TIMES = tuple(range(TS_BASE, TS_BASE + TS_COUNT))
LANES = tuple(range(LANE_BASE, LANE_BASE + LANE_COUNT))
TOUCHES = tuple(range(TOUCH_BASE, TOUCH_BASE + TOUCH_COUNT))
SHAPES = tuple(range(SLIDE_SHAPE_BASE, SLIDE_SHAPE_BASE + SLIDE_SHAPE_COUNT))
NOTE_TYPES = (NOTE_TAP, NOTE_TOUCH, NOTE_HOLD, NOTE_SLIDE)


def seconds_to_token(seconds: float) -> int:
    """将秒转换为 10ms 精度的 TS token，并限制在可表示范围内。"""
    raw = int(round(seconds * 100))
    if seconds > 0 and raw == 0:
        raw = 1
    return TS_BASE + max(0, min(TS_COUNT - 1, raw))


def token_to_seconds(token: int) -> float:
    if token not in TIMES:
        raise ValueError(f"不是 TS token: {token}")
    return (token - TS_BASE) / 100.0


def encode_note(note: "Note") -> list[int]:
    from maidata_parser import NoteType

    tokens: list[int] = []
    if note.type is NoteType.TAP:
        tokens.extend((NOTE_TAP, LANE_BASE + note.data.value - 1))
        if note.isBreak:
            tokens.append(IS_BREAK)
        if note.isEx:
            tokens.append(IS_EX)
    elif note.type is NoteType.HOLD:
        tokens.extend((NOTE_HOLD, LANE_BASE + note.data.lane.value - 1, seconds_to_token(note.data.holdTime)))
        if note.isBreak:
            tokens.append(IS_BREAK)
        if note.isEx:
            tokens.append(IS_EX)
    elif note.type is NoteType.TOUCH:
        tokens.extend((NOTE_TOUCH, TOUCH_BASE + note.data.Touch_area.value - 1))
        if note.data.isFirework:
            tokens.append(IS_FIREWORK)
    elif note.type is NoteType.TOUCH_HOLD:
        tokens.extend((NOTE_TOUCH, TOUCH_BASE + note.data.Touch_area.value - 1, seconds_to_token(max(0.0, note.data.holdTime))))
        if note.data.isFirework:
            tokens.append(IS_FIREWORK)
    elif note.type is NoteType.SLIDE:
        tokens.append(NOTE_SLIDE)
        if note.isBreak:
            tokens.append(IS_BREAK)
        if note.isEx:
            tokens.append(IS_EX)
        for segment_index, segment in enumerate(note.data):
            tokens.extend((
                SEGMENT_START,
                SLIDE_SHAPE_BASE + segment.shape.value - 1,
                LANE_BASE + segment.start_lane.value - 1,
                LANE_BASE + segment.end_lane.value - 1,
            ))
            if segment.middle_lane is not None:
                tokens.append(LANE_BASE + segment.middle_lane.value - 1)
            tokens.extend((seconds_to_token(segment.wait_duration), seconds_to_token(segment.trace_duration)))
            if segment.isClockwise is True:
                tokens.append(IS_CW)
            elif segment.isClockwise is False:
                tokens.append(IS_CCW)
            # 这些是整条 Slide 的头部属性，parser 会把状态复制到后续段，
            # token 只在第一段编码一次，避免产生多个星头或 no-head 标记。
            if segment_index == 0 and segment.isForceStar:
                tokens.append(IS_FORCE_STAR)
            if segment_index == 0 and segment.isFakeRotate:
                tokens.append(IS_FAKE_ROTATE)
            if segment.isSlideBreak:
                tokens.append(IS_SLIDE_BREAK)
            if segment_index == 0 and segment.isSlideNoHead:
                tokens.append(IS_SLIDE_NO_HEAD)
            tokens.append(SEGMENT_END)
    return tokens


def encode_frame(frame: "Frame", time_sec: float | None = None) -> list[int]:
    tokens = [FRAME_START, seconds_to_token(frame.time_sec if time_sec is None else time_sec)]
    for note in frame.notes:
        tokens.extend(encode_note(note))
    tokens.append(FRAME_END)
    return tokens


def decode_frames(tokens: list[int]) -> list["Frame"]:
    """把一个 segment 的 token 解码为相对时间 Frame；非法字段会警告并容错。"""
    from maidata_parser import Frame, Hold_data, Note, NoteType, SlideSegment, SlideShape, TapType, TouchType, Touch_data

    frames: list[Frame] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in (SOS, PAD):
            i += 1
            continue
        if tokens[i] == EOS:
            break
        if tokens[i] != FRAME_START:
            print(f"[tokenizer] 警告: 帧外未知 token {tokens[i]}（位置 {i}），跳过")
            i += 1
            continue
        i += 1
        if i < len(tokens) and tokens[i] in TIMES:
            time_sec = token_to_seconds(tokens[i])
            i += 1
        else:
            actual = tokens[i] if i < len(tokens) else "序列结束"
            print(f"[tokenizer] 警告: FRAME_START 后缺少有效时间戳（位置 {i}，实际 {actual}），使用 0 秒")
            time_sec = 0.0

        notes: list[Note] = []
        while i < len(tokens) and tokens[i] != FRAME_END:
            token = tokens[i]
            if token == NOTE_TAP:
                i += 1
                if i < len(tokens) and tokens[i] in LANES:
                    lane = TapType(tokens[i] - LANE_BASE + 1)
                else:
                    actual = tokens[i] if i < len(tokens) else "序列结束"
                    print(f"[tokenizer] 警告: TAP 缺少有效轨道（位置 {i}，实际 {actual}），使用 LANE1")
                    lane = TapType.LANE1
                i += int(i < len(tokens))
                is_break = is_ex = False
                while i < len(tokens) and tokens[i] in (IS_BREAK, IS_EX):
                    is_break |= tokens[i] == IS_BREAK
                    is_ex |= tokens[i] == IS_EX
                    i += 1
                notes.append(Note(NoteType.TAP, lane, isBreak=is_break, isEx=is_ex))
            elif token == NOTE_HOLD:
                i += 1
                if i < len(tokens) and tokens[i] in LANES:
                    lane = TapType(tokens[i] - LANE_BASE + 1)
                else:
                    actual = tokens[i] if i < len(tokens) else "序列结束"
                    print(f"[tokenizer] 警告: HOLD 缺少有效轨道（位置 {i}，实际 {actual}），使用 LANE1")
                    lane = TapType.LANE1
                i += int(i < len(tokens))
                if i < len(tokens) and tokens[i] in TIMES:
                    hold_sec = token_to_seconds(tokens[i])
                    i += 1
                else:
                    actual = tokens[i] if i < len(tokens) else "序列结束"
                    print(f"[tokenizer] 警告: HOLD 缺少有效持续时间（位置 {i}，实际 {actual}），使用 0 秒")
                    hold_sec = 0.0
                is_break = is_ex = False
                while i < len(tokens) and tokens[i] in (IS_BREAK, IS_EX):
                    is_break |= tokens[i] == IS_BREAK
                    is_ex |= tokens[i] == IS_EX
                    i += 1
                notes.append(Note(NoteType.HOLD, Hold_data(lane, hold_sec), isBreak=is_break, isEx=is_ex))
            elif token == NOTE_TOUCH:
                i += 1
                if i < len(tokens) and tokens[i] in TOUCHES:
                    area = TouchType(tokens[i] - TOUCH_BASE + 1)
                else:
                    actual = tokens[i] if i < len(tokens) else "序列结束"
                    print(f"[tokenizer] 警告: TOUCH 缺少有效区域（位置 {i}，实际 {actual}），使用 A1")
                    area = TouchType.A1
                i += int(i < len(tokens))
                hold_sec = token_to_seconds(tokens[i]) if i < len(tokens) and tokens[i] in TIMES else 0.0
                i += int(i < len(tokens) and tokens[i] in TIMES)
                firework = i < len(tokens) and tokens[i] == IS_FIREWORK
                i += int(firework)
                note_type = NoteType.TOUCH_HOLD if hold_sec > 0 else NoteType.TOUCH
                notes.append(Note(note_type, Touch_data(area, isFirework=firework, holdTime=hold_sec)))
            elif token == NOTE_SLIDE:
                i += 1
                note_break = note_ex = False
                while i < len(tokens) and tokens[i] in (IS_BREAK, IS_EX):
                    note_break |= tokens[i] == IS_BREAK
                    note_ex |= tokens[i] == IS_EX
                    i += 1
                segments: list[SlideSegment] = []
                while i < len(tokens) and tokens[i] == SEGMENT_START:
                    i += 1
                    if i < len(tokens) and tokens[i] in SHAPES:
                        shape = SlideShape(tokens[i] - SLIDE_SHAPE_BASE + 1)
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少有效形状（位置 {i}，实际 {actual}），使用 Line")
                        shape = SlideShape.Line
                    i += int(i < len(tokens))
                    if i < len(tokens) and tokens[i] in LANES:
                        start_lane = TapType(tokens[i] - LANE_BASE + 1)
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少有效起点轨道（位置 {i}，实际 {actual}），使用 LANE1")
                        start_lane = TapType.LANE1
                    i += int(i < len(tokens))
                    if i < len(tokens) and tokens[i] in LANES:
                        end_lane = TapType(tokens[i] - LANE_BASE + 1)
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少有效终点轨道（位置 {i}，实际 {actual}），使用 LANE1")
                        end_lane = TapType.LANE1
                    i += int(i < len(tokens))
                    middle_lane = None
                    if shape is SlideShape.GrandV and i < len(tokens) and tokens[i] in LANES:
                        middle_lane = TapType(tokens[i] - LANE_BASE + 1)
                        i += 1
                    elif shape is SlideShape.GrandV:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: GrandV 缺少有效中间轨道（位置 {i}，实际 {actual}）")
                    if i < len(tokens) and tokens[i] in TIMES:
                        wait_sec = token_to_seconds(tokens[i])
                        i += 1
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少有效等待时间（位置 {i}，实际 {actual}），使用 0 秒")
                        wait_sec = 0.0
                    if i < len(tokens) and tokens[i] in TIMES:
                        trace_sec = token_to_seconds(tokens[i])
                        i += 1
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少有效滑动时间（位置 {i}，实际 {actual}），使用 0 秒")
                        trace_sec = 0.0
                    clockwise = None
                    force_star = fake_rotate = slide_break = no_head = False
                    while i < len(tokens) and tokens[i] != SEGMENT_END:
                        if tokens[i] == IS_CW:
                            clockwise = True
                        elif tokens[i] == IS_CCW:
                            clockwise = False
                        elif tokens[i] == IS_FORCE_STAR:
                            force_star = True
                        elif tokens[i] == IS_FAKE_ROTATE:
                            fake_rotate = True
                        elif tokens[i] == IS_SLIDE_BREAK:
                            slide_break = True
                        elif tokens[i] == IS_SLIDE_NO_HEAD:
                            no_head = True
                        else:
                            print(f"[tokenizer] 警告: Slide 段出现未知属性 token {tokens[i]}（位置 {i}）")
                            break
                        i += 1
                    if i < len(tokens) and tokens[i] == SEGMENT_END:
                        i += 1
                    else:
                        actual = tokens[i] if i < len(tokens) else "序列结束"
                        print(f"[tokenizer] 警告: Slide 段缺少 SEGMENT_END（位置 {i}，实际 {actual}）")
                    segments.append(SlideSegment(shape, start_lane, end_lane, wait_sec, trace_sec, clockwise, middle_lane,
                                                 force_star, fake_rotate, slide_break, no_head))
                if not segments:
                    print(f"[tokenizer] 警告: SLIDE 没有有效段（位置 {i}）")
                notes.append(Note(NoteType.SLIDE, segments, isBreak=note_break, isEx=note_ex))
            else:
                print(f"[tokenizer] 警告: 帧内未知 token {tokens[i]}（位置 {i}），跳过")
                i += 1
        if i < len(tokens) and tokens[i] == FRAME_END:
            i += 1
        elif notes:
            actual = tokens[i] if i < len(tokens) else "序列结束"
            print(f"[tokenizer] 警告: 帧缺少 FRAME_END（位置 {i}，实际 {actual}）")
        if notes:
            frames.append(Frame(notes=tuple(notes), time_sec=time_sec))
    frames.sort(key=lambda frame: frame.time_sec)
    return frames


def rotate_token_id(token: int, steps: int) -> int:
    steps %= LANE_COUNT
    if token in LANES:
        return LANE_BASE + ((token - LANE_BASE + steps) % LANE_COUNT)
    if token in TOUCHES:
        offset = token - TOUCH_BASE
        if offset < 8:
            return TOUCH_BASE + ((offset + steps) % 8)
        if offset < 16:
            return TOUCH_BASE + 8 + ((offset - 8 + steps) % 8)
        if offset == 16:
            return token
        if offset < 25:
            return TOUCH_BASE + 17 + ((offset - 17 + steps) % 8)
        return TOUCH_BASE + 25 + ((offset - 25 + steps) % 8)
    return token


def rotate_tokens(tokens: list[int], steps: int) -> list[int]:
    return [rotate_token_id(token, steps) for token in tokens]


def _self_check() -> None:
    from contextlib import redirect_stdout
    from io import StringIO

    from maidata_parser import (
        Frame, Hold_data, Note, NoteType, SlideSegment, SlideShape, TapType,
        TouchType, Touch_data, compiler,
    )

    assert VOCAB_SIZE == 3072
    assert seconds_to_token(-1) == TS_BASE
    assert seconds_to_token(0.001) == TS_BASE + 1
    assert seconds_to_token(99) == TS_BASE + 2999

    notes = (
        Note(NoteType.TAP, TapType.LANE8, isBreak=True, isEx=True),
        Note(NoteType.HOLD, Hold_data(TapType.LANE2, 1.23)),
        Note(NoteType.TOUCH, Touch_data(TouchType.C, isFirework=True)),
        Note(NoteType.TOUCH_HOLD, Touch_data(TouchType.D4, holdTime=0.45)),
        Note(NoteType.SLIDE, [
            SlideSegment(
                SlideShape.GrandV, TapType.LANE1, TapType.LANE5, 0.2, 0.8,
                middle_lane=TapType.LANE3, isForceStar=True, isSlideNoHead=True,
            ),
        ], isBreak=True),
    )
    decoded = decode_frames([SOS, *encode_frame(Frame(notes, 12.34)), EOS, PAD])
    assert len(decoded) == 1
    assert decoded[0].time_sec == 12.34
    assert decoded[0].notes == notes

    note = Note(NoteType.HOLD, Hold_data(TapType.LANE3, 0.5), isEx=True)
    parser = compiler()
    assert parser._ts_token(0.5) == seconds_to_token(0.5)
    assert parser._encode_note_tokens(note) == encode_note(note)
    tokens = [FRAME_START, seconds_to_token(1.0), *encode_note(note), FRAME_END]
    assert parser._parse_token_segment(tokens) == decode_frames(tokens)

    tokens = [SOS, TOUCH_BASE + 16, *LANES, EOS]
    rotated = tokens
    for _ in range(8):
        rotated = rotate_tokens(rotated, 1)
    assert rotated == tokens
    assert rotate_token_id(TOUCH_BASE + 16, 3) == TOUCH_BASE + 16

    output = StringIO()
    malformed = [
        9999,
        FRAME_START, NOTE_TAP, 9999,
        NOTE_HOLD, LANE_BASE, FRAME_END,
        FRAME_START, TS_BASE, NOTE_TOUCH, 9999, FRAME_END,
        FRAME_START, TS_BASE, NOTE_SLIDE, SEGMENT_START, 9999, 9999, 9999,
    ]
    with redirect_stdout(output):
        decode_frames(malformed)
    warnings = output.getvalue()
    for expected in (
        "帧外未知 token",
        "FRAME_START 后缺少有效时间戳",
        "TAP 缺少有效轨道",
        "HOLD 缺少有效持续时间",
        "TOUCH 缺少有效区域",
        "Slide 段缺少有效形状",
        "Slide 段缺少 SEGMENT_END",
        "帧缺少 FRAME_END",
    ):
        assert expected in warnings, f"缺少警告: {expected}\n{warnings}"
    print("[tokenizer] 自检通过")


if __name__ == "__main__":
    _self_check()
