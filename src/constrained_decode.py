"""maimai token 结构化解码规范。

约束目标
========

约束器应把模型输出限制在项目 token 编码能够表达、parser 能够解析、且符合普通
Master 谱面物理语义的范围内。约束分为四层：序列结构、单音符结构、同一逻辑帧
资源冲突、跨帧持续资源占用。硬约束只能用于确定非法或训练数据清洗后已无反例的
情况，不能把低频但合法的谱面写法当作非法。

时间统一使用 TS token 的 10ms 精度。所有持续区间采用半开区间 ``[start, end)``：
音符开始时间小于 ``end`` 时资源仍被占用，恰好等于 ``end`` 时已经释放，不添加
10ms、50ms 或其他人为冷却时间。

一、序列与帧结构
----------------

1. 完整序列只能是 ``SOS (FRAME)* EOS``；SOS 只能位于开头且只能出现一次，EOS
   只能位于完整序列末尾，PAD 不能出现在有效序列中间。
2. SOS 后只能输出 FRAME_START 或 EOS；FRAME_END 后只能输出下一个 FRAME_START
   或 EOS。
3. FRAME_START 后必须输出帧时间戳 TS，随后必须至少输出一个完整 Note，最后输出
   FRAME_END；不允许空帧。
4. 帧时间戳不能倒退，但允许相等。训练数据存在原始时间不同、量化到同一个 10ms
   TS 的多个 frame；相同 TS 在语义检查时必须视为同一个逻辑帧，累计检查重复和
   资源冲突，不能简单禁止第二个同 TS frame。
5. malformed prefix 不能靠在末尾追加 token“修复”。例如已经出现
   ``NOTE_TAP FRAME_END`` 时缺失的 LANE 无法再插入，应返回空 allowed set。
   FRAME_END 后的垃圾 token、重复 SOS、内部 EOS 等也必须判为死路，不能静默忽略。
6. 每个物理 frame 最多 33 个 Note；TAP/HOLD/SLIDE 合计最多 4 个，TOUCH 不计入
   action 上限。这两个值是当前普通 Master 训练集观察上限，不是 simai 语法常量。
   达到上限后不能再开始新 Note，但已经开始的 Note 必须允许合法闭合，不能生成
   半个 Note 后强制 FRAME_END。

二、单音符规范
--------------

TAP
~~~

``NOTE_TAP -> LANE -> [IS_BREAK] -> [IS_EX]``。

- IS_BREAK、IS_EX 各最多一次，并按编码器固定顺序出现。
- ``IS_BREAK + IS_EX`` 是合法组合，不能设为互斥。

HOLD
~~~~

``NOTE_HOLD -> LANE -> TS(duration) -> [IS_BREAK] -> [IS_EX]``。

- duration 使用 TS_0..TS_2999。普通谱面规范应要求 TS_1..TS_2999，但训练集中仍有
  零时长 HOLD；在将它们归一化为 TAP 或排除前，不能直接屏蔽 TS_0。
- 属性去重、顺序及组合规则与 TAP 相同。

TOUCH / TOUCH_HOLD
~~~~~~~~~~~~~~~~~~

``NOTE_TOUCH -> TOUCH_AREA -> [TS(duration)] -> [IS_FIREWORK]``。

- 不输出 duration 表示普通 TOUCH；输出 duration 表示 TOUCH_HOLD。
- canonical token 中 TOUCH_HOLD duration 必须为 TS_1..TS_2999；``TS_0`` 与省略
  duration 都表示普通 TOUCH，约束器应优先只保留省略 duration 的规范形式。
- IS_FIREWORK 可用于普通 TOUCH 和 TOUCH_HOLD，也不限于 C 区。
- TouchHold 不限于 C 区，不能因训练数据中非 C 样本较少而禁止。

SLIDE NOTE
~~~~~~~~~~

``NOTE_SLIDE -> [IS_BREAK] -> [IS_EX] -> SEGMENT+``。

- Slide 必须至少包含一个完整 segment，不允许 ``NOTE_SLIDE FRAME_END`` 空 Slide。
  当前训练源仍有一个 parser 异常空 Slide，启用该硬约束前应先清洗该样本。
- Note 级 IS_BREAK 和 IS_EX 可同时存在；Note 级 IS_BREAK 与 segment 级
  IS_SLIDE_BREAK 也不互斥。
- 同一 Slide 可以有多个 segment，不能把 segment 数量当作 action note 数量。

SLIDE SEGMENT
~~~~~~~~~~~~~

结构为：

``SEGMENT_START -> SHAPE -> start LANE -> end LANE``
``[GrandV middle LANE] -> TS(wait) -> TS(trace) -> [属性] -> SEGMENT_END``。

- GrandV 必须有 middle lane，其他 shape 不得有 middle lane；GrandV 的 middle 必须
  不同于 start 和 end。
- wait 和 trace 不能同时为 TS_0。trace 单独为 TS_0 必须允许，训练数据中的链式
  Slide 中间段会使用它；不能笼统要求 trace 大于零。
- IS_CW 与 IS_CCW 仅适用于 Circle，且互斥；Circle 两者都不输出表示 ``^`` 自动
  选择最短方向。非 Circle 不能输出方向属性。
- segment 属性各最多一次，并按编码器顺序出现：方向、IS_FORCE_STAR、
  IS_FAKE_ROTATE、IS_SLIDE_BREAK、IS_SLIDE_NO_HEAD。
- IS_FORCE_STAR、IS_FAKE_ROTATE、IS_SLIDE_NO_HEAD 属于 Slide 头部语义，原则上只应
  出现在第一段。当前 parser 会在一个特殊谱面中把 no-head 复制到第二段，清洗或
  修正编码前不能直接屏蔽该训练标签。
- 当前 parser 会把整条 Slide 的 Break 复制到多个 segment，训练中存在大量非末段
  IS_SLIDE_BREAK。不能在未统一 parser/token 语义前强制“只有末段可 Break”。

三、Slide 几何约束
------------------

令 ``delta = (end - start) mod 8``，lane 使用 0..7 计算：

- Line：只允许 delta 为 2、3、4、5、6；不允许同轨或相邻轨退化直线。
- S、Z、Wifi：必须 ``delta == 4``，即终点位于正对面。
- V、GrandV：必须 ``delta != 0``。
- Circle、P、Q、PP、QQ：允许 delta 0..7；训练数据明确存在 start=end 的合法绕行
  或旋转写法，不能统一禁止所有 Slide 同起终点。
- GrandV middle 不能等于 start/end；暂不强制 middle 必须为 start±2，因为训练中
  存在少量其他合法或待核查写法。

多段 Slide 的第二段及以后，start lane 只能是：

- 前一段的 end lane：链式 Slide；或
- 第一段的 start lane：``*`` 多重/分叉 Slide。

不能要求所有后续段都连接前段终点，否则会误杀 ``*``；也不能允许任意第三个起点，
因为当前 parser/generator 无法稳定 round-trip 这种断开的 segment。

允许同起点多条不同路线、路线汇合、相同局部 segment，以及一个 Slide Note 内的
多个同起点 segment。只因 segment 重复或数量较多就硬过滤会误杀合法谱面。

四、同一逻辑帧的资源约束
------------------------

同一 TS 下，即使 token 使用多个 FRAME 表示，也要合并后执行以下规则：

- 同一 lane 不允许重复 TAP。
- 同一 lane 不允许 TAP 与 HOLD 同时开始。
- 同一 lane 不允许多个 HOLD 同时开始。
- 同一归一化 Touch area 不允许重复 TOUCH、重复 TOUCH_HOLD，或 TOUCH 与
  TOUCH_HOLD 同时开始。C、C1、C2 在 parser 中都归一化为同一个 C area。
- 禁止完全相同的独立 Note 重复；但不能因此禁止同起点、不同路线的多 Slide。
- 同 lane TAP/HOLD 与普通有头 Slide 的组合通常是资源冲突，但 no-head Slide 可以
  与独立 TAP/HOLD 组合；由于 IS_SLIDE_NO_HEAD 在 segment 尾部才出现，实施该规则
  时不能在知道 no-head 之前过早屏蔽合法结构。
- 同起点的多个 Slide 必须允许；训练数据中存在合法实例。

五、跨帧持续资源占用
--------------------

HOLD 占用普通 lane。若 lane L 上存在 HOLD ``[start, end)``：

- ``frame_time < end`` 时，禁止 TAP(L)、HOLD(L)，以及第一段从 L 开始的 SLIDE。
- ``frame_time == end`` 时允许同 lane 新音符，不设置额外冷却。
- 其他 lane 的 TAP/HOLD/SLIDE 和所有 TOUCH 仍然允许。
- decoder 的 6 秒 token 前缀中可能已有尚未结束的 HOLD，状态必须从完整前缀恢复，
  不能只追踪当前目标区新生成的 HOLD。

TOUCH_HOLD 占用具体 Touch area。若 area A 上存在 TOUCH_HOLD ``[start, end)``：

- ``frame_time < end`` 时禁止 TOUCH(A) 和 TOUCH_HOLD(A)。
- ``frame_time == end`` 时允许重新进入。
- 只占用具体 area，不得把整个 A/B/C/D/E 大区一起锁定。

六、明确不能采用的粗粒度规则
----------------------------

- 不能禁止 HOLD 持续期间的所有其他音符；不同 lane 音符非常常见。
- 不能禁止 HOLD 结束瞬间接音符，也不能添加任意冷却时间。
- 不能禁止 Slide 活跃期间出现其他音符，或禁止使用 Slide 起终点 lane。
- 不能禁止同起点多 Slide、多段 Slide、路线分叉或汇合。
- 不能禁止所有 start=end Slide；Circle/P/Q/PP/QQ 有真实合法样本。
- 不能禁止 trace=TS_0、Break+EX、Circle 无方向属性、非 C TouchHold，或大量不同
  area 的 TOUCH 同时出现。
- 不能把每帧 action 上限降为 2，也不能给 TOUCH 设置很低的总数上限。

七、当前实现状态
----------------

当前文件已经实现基础序列、Note 字段、时间非递减、属性去重、CW/CCW 互斥，以及
33 Note / 4 action 的观察上限；尚未完整实现上文的 canonical 属性顺序、Slide
几何、多段连接、同帧资源冲突、HOLD/TOUCH_HOLD 跨帧占用和严格 malformed prefix
校验。修改状态机后必须对全部训练 token 逐 token 重放，默认硬约束应达到
``constraint_violations = 0``；有反例的规则必须先清洗数据或修正 parser，不能仅在
推理端强行屏蔽。
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
