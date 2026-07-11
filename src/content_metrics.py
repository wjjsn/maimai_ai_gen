from maidata_parser import EOS, PAD, SOS, NoteType, compiler


def content_match_counts(generated, truth, tolerance_sec=0.01):
    pred_events = note_events(generated)
    target_events = note_events(truth)
    return _match_events(pred_events, target_events, tolerance_sec)


def content_match_frame_counts(generated_frames, target_frames, tolerance_sec=0.01):
    return _match_events(frame_events(generated_frames), frame_events(target_frames), tolerance_sec)


def _match_events(pred_events, target_events, tolerance_sec):
    used = [False] * len(target_events)
    tp = 0
    for pred_time, pred_type in pred_events:
        best_i = -1
        best_dt = tolerance_sec
        for i, (target_time, target_type) in enumerate(target_events):
            if used[i] or pred_type != target_type:
                continue
            dt = abs(pred_time - target_time)
            if dt <= best_dt + 1e-9:
                best_i = i
                best_dt = dt
        if best_i >= 0:
            used[best_i] = True
            tp += 1
    return tp, len(pred_events), len(target_events)


def note_events(tokens):
    body = [t for t in tokens if t not in (PAD, SOS, EOS)]
    frames = compiler()._parse_token_segment(body)
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


if __name__ == "__main__":
    _self_check()
    print("内容指标自检通过")
