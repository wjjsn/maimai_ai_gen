from maidata_parser import NoteType
from tokenizer import EOS, PAD, SOS, decode_frames


def content_match_counts(generated, truth, tolerance_sec=0.01):
    pred_events = note_events(generated)
    target_events = note_events(truth)
    return _match_events(pred_events, target_events, tolerance_sec)


def content_match_frame_counts(generated_frames, target_frames, tolerance_sec=0.01):
    return _match_events(frame_events(generated_frames), frame_events(target_frames), tolerance_sec)


def _match_events(pred_events, target_events, tolerance_sec):
    """按类型分组后在时间轴上贪心匹配，得到最大命中数。"""
    pred_by_type = {}
    target_by_type = {}
    for time, kind in pred_events:
        pred_by_type.setdefault(kind, []).append(time)
    for time, kind in target_events:
        target_by_type.setdefault(kind, []).append(time)
    tp = 0
    for kind, pred_times in pred_by_type.items():
        target_times = target_by_type.get(kind, [])
        pred_times.sort()
        target_times.sort()
        pred_i = 0
        target_i = 0
        while pred_i < len(pred_times) and target_i < len(target_times):
            pred_time = pred_times[pred_i]
            target_time = target_times[target_i]
            if pred_time < target_time - tolerance_sec - 1e-9:
                pred_i += 1
            elif target_time < pred_time - tolerance_sec - 1e-9:
                target_i += 1
            else:
                tp += 1
                pred_i += 1
                target_i += 1
    return tp, len(pred_events), len(target_events)


def note_events(tokens):
    body = [t for t in tokens if t not in (PAD, SOS, EOS)]
    frames = decode_frames(body)
    return frame_events(frames)


def frame_events(frames):
    return [
        (frame.time_sec, _note_type(note))
        for frame in frames
        for note in frame.notes
    ]


def _duration_key(seconds):
    return round(seconds * 100)


def _note_type(note):
    if note.type is NoteType.HOLD:
        return note.type, _duration_key(note.data.holdTime)
    if note.type is NoteType.TOUCH_HOLD:
        return note.type, _duration_key(note.data.holdTime)
    if note.type is NoteType.SLIDE:
        return note.type, tuple(
            (
                segment.shape,
                _duration_key(segment.wait_duration),
                _duration_key(segment.trace_duration),
                segment.isClockwise,
                segment.isForceStar,
                segment.isFakeRotate,
                segment.isSlideNoHead,
            )
            for segment in note.data
        )
    return note.type


def _self_check():
    from maidata_parser import Hold_data, Note, SlideSegment, SlideShape, TapType

    tap_plain = Note(NoteType.TAP, TapType.LANE1)
    tap_attr = Note(NoteType.TAP, TapType.LANE8, isBreak=True, isEx=True)
    assert _note_type(tap_plain) == _note_type(tap_attr)

    hold_short = Note(NoteType.HOLD, Hold_data(TapType.LANE1, 0.1))
    hold_short_attr = Note(NoteType.HOLD, Hold_data(TapType.LANE8, 0.1), isBreak=True, isEx=True)
    hold_long = Note(NoteType.HOLD, Hold_data(TapType.LANE8, 2.0), isBreak=True)
    assert _note_type(hold_short) == _note_type(hold_short_attr)
    assert _note_type(hold_short) != _note_type(hold_long)

    line = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE1, TapType.LANE3, 0.2, 0.5)])
    line_rotated = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE4, TapType.LANE6, 0.2, 0.5)])
    circle = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Circle, TapType.LANE1, TapType.LANE3)])
    line_ccw = Note(NoteType.SLIDE, [SlideSegment(SlideShape.Line, TapType.LANE1, TapType.LANE3, 0.2, 0.5, False)])
    assert _note_type(line) == _note_type(line_rotated)
    assert _note_type(line) != _note_type(circle)
    assert _note_type(line) != _note_type(line_ccw)

    tap = _note_type(tap_plain)
    assert _match_events([(1.01, tap), (1.01, tap)], [(1.0, tap)], 0.01) == (1, 2, 1)
    assert _match_events([(1.011, tap)], [(1.0, tap)], 0.01) == (0, 1, 1)
    assert _match_events([(0.010, tap), (0.020, tap)], [(0.000, tap), (0.019, tap)], 0.011) == (2, 2, 2)


if __name__ == "__main__":
    _self_check()
    print("内容指标自检通过")
