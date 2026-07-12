"""maimai token 结构化解码规范。

约束目标
========

约束器应把模型输出限制在项目 token 编码能够表达、parser 能够解析、且符合高质量
普通 Master 谱面物理语义的范围内。约束分为序列结构、单音符结构、同一逻辑帧
资源冲突、跨帧持续资源占用和普通谱面质量限制。

本规范不以“兼容当前全部训练标签”为目标。训练源中可能混有宴会场风格、特殊
谱面、转换错误、parser 误解析和物理上不合理的配置。规则有少量反例不代表规则
错误；对于高置信异常，宁可排除数据，也不能为了保留少数脏标签放宽生成语言。

训练数据采用整曲排除：一首歌的 Master 只要命中任一默认硬规则，整首歌都不得进入
训练集、验证集和整曲生成指标。禁止只删除单个 Note 或 frame，因为局部删除会破坏
节奏结构、Slide/Hold 上下文、滑窗 decoder 前缀和整曲真值的一致性。所有排除必须
记录歌曲路径、规则编号、时间、lane/area 和相关 Note，供离线审计。

训练缓存构建、离线数据审计和推理结构化解码必须共享同一套规则及时间量化方式，
不能分别维护近似实现。缓存规则或 token 语义变化时必须提升 CACHE_VERSION，并对
全部训练源重新审计和重建缓存。

时间统一使用 TS token 的 10ms 精度：``1cs = 10ms``。持续资源的实际占用区间为
半开区间 ``[start, end)``；此外，为普通谱面手感增加 ``50ms = 5cs`` 的同位置释放
冷却。也就是说，HOLD/TOUCH_HOLD 在 ``end`` 时结束物理占用，但同 lane/area 的新
独立输入只有在 ``frame_time > end + 5cs`` 时才允许。``frame_time == end`` 仍属于
冷却冲突，不再作为合法的瞬间 release-tap 例外。

默认质量阈值
------------

- HOLD/TOUCH_HOLD 结束后同位置冷却：50ms，包含边界，即 ``0 <= gap <= 5cs`` 禁止。
- 同 lane 两个独立 press 的最小间隔：30ms；``gap < 3cs`` 判为硬冲突。
- HOLD/TOUCH_HOLD duration 必须为正，不能是 TS_0；现有零时长标签应修正 parser、
  归一化或整曲排除，不能继续作为普通 HOLD 学习。
- duration 不能超过 TS_2999；编码时不得静默 clamp 超范围值。
- 30ms 到 50ms 的同 lane press、极短 HOLD、高密度 jack 和异常长 HOLD 属于质量
  审计候选。它们可以升级为整曲排除规则，但不能悄悄修改单个 Note。

规则分级
--------

- 默认硬规则：结构确定非法、物理资源冲突，或统计上高度集中于特殊/异常谱面的
  配置。训练和推理都必须拒绝，旧数据有反例时排除整首歌。
- 质量审计规则：尚不能证明普遍非法，但不符合普通 Master 目标。先报告命中规模和
  典型谱面，经确认后整体升级为硬规则。
- 明确保留规则：真实普通谱面中广泛存在的写法，不能因实现方便而粗粒度禁止。

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
6. 状态机不得暴露已知必然走入死路的下一 token。无可用 frame 时间或 frame 不允许
   生成任何 Note 时，序列边界只能输出 EOS，不能先允许 FRAME_START 再在后续返回空
   allowed set。所有 Touch area 都不可用时，也不能继续暴露 NOTE_TOUCH。
7. 每个物理 frame 最多 33 个 Note；同一逻辑 TS 最多两个独立 press。独立 press 是
   TAP、HOLD_START 和有头 SLIDE_HEAD；TOUCH、TOUCH_HOLD 与 no-head Slide 不计入。
   达到两个 press 后仍可生成 TOUCH 或 no-head Slide，但不能生成新的 TAP、HOLD 或
   有头 Slide。已经开始的 Note 必须允许按合法形式闭合，不能生成半个 Note 后强制
   FRAME_END。

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

- duration 必须使用 TS_1..TS_2999。TS_0 HOLD 没有持续动作，不能进入普通 Master
  训练目标；若源谱使用 pseudo-HOLD，应在 parser/清洗阶段归一化为 TAP，不能要求
  解码器保留脏标签。
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
  当前训练源中的 parser 异常空 Slide 应导致整曲排除，不能成为放宽规则的理由。
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
- IS_FORCE_STAR、IS_FAKE_ROTATE、IS_SLIDE_NO_HEAD 属于 Slide 头部语义，只应出现
  在第一段。parser 若把头部属性复制到后续段，应修正 parser 或排除整曲，不能要求
  解码器接受非规范 token。
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

完全重复检查发生在 Slide Note 边界，而不是 segment 边界。一个新 Slide 的当前
segment 即使与已有 Slide 完全相同，也必须允许 SEGMENT_END，因为它还可以继续增加
segment 形成不同路线；只有当前完整 Slide Note 已与同一逻辑帧中的已有 Note 重复
时，才禁止结束 Note，并只允许继续输出 SEGMENT_START。

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
- ``end <= frame_time <= end + 5cs`` 时仍处于释放冷却，禁止同 lane TAP、HOLD 和
  有头 Slide；只有 ``frame_time > end + 5cs`` 才允许同 lane 新独立 press。
- Slide 是否产生独立 press 由第一段 IS_SLIDE_NO_HEAD 决定。有头 Slide 受 lane
  占用与冷却限制；no-head Slide 不要求重新按下，不能无条件按有头 Slide 处理。
- 其他 lane 的 TAP/HOLD/SLIDE 和所有 TOUCH 仍然允许。
- decoder 的 6 秒 token 前缀中可能已有尚未结束的 HOLD，状态必须从完整前缀恢复，
  不能只追踪当前目标区新生成的 HOLD。
- 同 lane 两个正时长 HOLD 的占用区间不得重叠。

TOUCH_HOLD 占用具体 Touch area。若 area A 上存在 TOUCH_HOLD ``[start, end)``：

- ``frame_time < end`` 时禁止 TOUCH(A) 和 TOUCH_HOLD(A)。
- ``end <= frame_time <= end + 5cs`` 时仍处于释放冷却，继续禁止同 area TOUCH 和
  TOUCH_HOLD；只有 ``frame_time > end + 5cs`` 才允许重新进入。
- 只占用具体 area，不得把整个 A/B/C/D/E 大区一起锁定。

六、普通谱面质量规则
--------------------

独立 press 定义为 TAP、HOLD_START 和有头 SLIDE_HEAD。一个 Slide Note 即使包含多个
``*`` 分支或链式 segment，也只按一个 Slide head 计算；no-head Slide 不算新 press。

默认硬规则：

- 同一逻辑帧、同一 lane 的独立 press 数不能超过 1。
- 同 lane 相邻独立 press 的间隔不能小于 30ms，即 ``gap < 3cs`` 时整曲排除。
- HOLD 活跃期间以及结束后 50ms 冷却内，不得出现同 lane 独立 press。
- TOUCH_HOLD 活跃期间以及结束后 50ms 冷却内，不得出现同 area TOUCH/TOUCH_HOLD。
- 同 lane HOLD 区间不得重叠，同 area TOUCH_HOLD 区间不得重叠。
- 同一逻辑帧不得有完全相同的独立 Note；同 Touch area 不得重复或同时出现 TOUCH
  与 TOUCH_HOLD。
- HOLD、TOUCH_HOLD duration 必须为 TS_1..TS_2999；Slide wait/trace 不得同时为
  TS_0；任何 duration 超出词表范围都应拒绝，不能 clamp 后继续训练。
- 空 Slide、断开的多段 Slide、非法 Shape 几何、非 Circle 方向属性、属性倒序或
  重复、malformed frame/sequence 都属于整曲排除条件。
- parser 失败、无法稳定 round-trip、Master 等级元数据明显异常或被标记为特殊谱面
  时，默认不进入普通 Master 数据集。

质量审计候选，不在没有统计报告时直接扩大为硬规则：

- ``30ms <= gap < 50ms`` 的同 lane 独立 press。
- ``0 < HOLD duration < 50ms`` 的极短 HOLD。
- 250ms 内同 lane 独立 press 至少 5 次，或 500ms 内至少 9 次。
- HOLD duration 大于 10 秒；大于 15 秒时应优先人工复核。
- Slide 的重复完整路线、异常多 segment、异常长 wait/trace，以及 Slide 结束后极短
  间隔的端点 lane 输入。这些不能仅靠端点共享就判非法，必须结合实际头部和轨迹。

已知高优先级异常模式包括：长 HOLD 内同 lane TAP、HOLD 结束点瞬间接同 lane TAP、
连续 30ms 到 50ms 极短 HOLD、低于 30ms 的同 lane Slide head，以及零时长 HOLD。
这类模式即使只集中于少数歌曲，也应排除歌曲，而不是作为“合法少数派”放行。

七、明确不能采用的粗粒度规则
----------------------------

- 不能禁止 HOLD 持续期间的所有其他音符；不同 lane 音符非常常见。
- 不能把 50ms 冷却扩大到所有 lane；它只限制刚释放的同 lane/area。
- 不能未经统计把冷却提高到 80ms 或 100ms。现有分布在 80ms 附近开始大量覆盖正常
  高 BPM 谱面，阈值变化必须重新审计整库影响。
- 不能禁止 Slide 活跃期间出现其他音符，或禁止使用 Slide 起终点 lane。
- 不能禁止同起点多 Slide、多段 Slide、路线分叉或汇合。
- 不能禁止所有 start=end Slide；Circle/P/Q/PP/QQ 有真实合法样本。
- 不能禁止 trace=TS_0、Break+EX、Circle 无方向属性、非 C TouchHold，或大量不同
  area 的 TOUCH 同时出现。
- 不能把两个 press 的上限错误套到 TOUCH、TOUCH_HOLD、no-head Slide 或 Slide 的
  后续 segment，也不能给 TOUCH 设置很低的总数上限。

八、数据清洗与验证要求
----------------------

1. 离线审计以整首歌为单位输出通过/拒绝和全部原因，不能遇到第一条错误就丢失其他
   诊断信息。
2. 拒绝记录至少包含规则 ID、歌曲相对路径、level、绝对时间、量化时间、lane/area、
   Note 类型、持续时间和相关前后事件。
3. 训练集和验证集只能从审计通过的歌曲中划分；同一首违规歌曲不能只从训练集删除
   却留在验证或整曲生成评估中。
4. 旋转增强不改变规则结果。原谱通过时，8 个旋转版本都必须通过；原谱违规时不得
   通过旋转掩盖 lane 冲突。
5. 滑窗前缀必须携带足够状态，使跨窗口 HOLD/TOUCH_HOLD 占用和 50ms 冷却在训练、
   teacher forcing、正式递推推理中完全一致。
6. 每次修改规则都要报告：扫描歌曲数、通过/拒绝歌曲数、各规则命中歌曲数和事件数、
   典型样本，以及对窗口数和 train/val 歌曲数的影响。
7. 约束器的自检应同时包含合法边界和拒绝边界：HOLD 结束后 50ms 必须拒绝，60ms
   必须允许；不同 lane 在 HOLD 期间必须允许；no-head Slide 不能误算独立 press。

九、当前实现状态
----------------

当前文件已经用严格前缀重放状态机实现：SOS/EOS/frame 结构、canonical Note 字段和
属性顺序、时间非递减、同 TS 逻辑帧合并、33 Note / 2 press 上限、Slide 几何与
多段连接、同帧 lane/Touch 资源冲突、HOLD/TOUCH_HOLD 跨帧占用、50ms 释放冷却、
30ms 独立 press 最小间隔、no-head Slide 例外，以及 malformed prefix 死路处理。
序列边界只在至少存在一个合法 frame 时间且 frame 可容纳 Note 时暴露 FRAME_START；
重复 Slide 在完整 Note 边界强制继续扩展，不会错误禁止可形成不同路线的相同局部
segment；耗尽的 Touch 类型不会继续作为候选暴露。

尚未实现的是 chart_cache 构建前的整曲质量审计和自动排除报告。因此当前推理解码已
执行严格规范，但旧缓存仍可能包含会被新规则拒绝的歌曲。后续应让缓存构建先按同一
规则排除整曲，再要求清洗后的训练 token 达到 ``constraint_violations = 0``。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tokenizer import (
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
    LANES,
    NOTE_TYPES,
    SHAPES,
    TIMES,
    TOUCHES,
    encode_frame,
    encode_note,
)

if TYPE_CHECKING:
    from maidata_parser import Frame


GRAND_V = SLIDE_SHAPE_BASE + 3
LINE = SLIDE_SHAPE_BASE
CIRCLE = SLIDE_SHAPE_BASE + 1
V_SHAPE = SLIDE_SHAPE_BASE + 2
S_SHAPE = SLIDE_SHAPE_BASE + 8
Z_SHAPE = SLIDE_SHAPE_BASE + 9
WIFI = SLIDE_SHAPE_BASE + 10
POSITIVE_TIMES = TIMES[1:]
RELEASE_COOLDOWN_CS = 5
MIN_PRESS_GAP_CS = 3


class _NeedToken(Exception):
    def __init__(self, allowed: tuple[int, ...]):
        self.allowed = allowed


class _InvalidPrefix(Exception):
    pass


@dataclass(frozen=True)
class _NoteRecord:
    signature: tuple
    press_lane: int | None = None
    hold: tuple[int, int] | None = None
    touch_hold: tuple[int, int] | None = None
    touch_area: int | None = None


@dataclass
class _ReplayState:
    frame_time: int | None = None
    previous_frame_time: int = 0
    logical_time: int | None = None
    logical_signatures: set[tuple] = field(default_factory=set)
    logical_press_lanes: set[int] = field(default_factory=set)
    logical_touch_areas: set[int] = field(default_factory=set)
    active_holds: dict[int, int] = field(default_factory=dict)
    active_touch_holds: dict[int, int] = field(default_factory=dict)
    last_press_time: dict[int, int] = field(default_factory=dict)
    frame_notes: int = 0
    frame_actions: int = 0

    def begin_frame(self, time_cs: int) -> None:
        if time_cs < self.previous_frame_time:
            raise _InvalidPrefix
        self.previous_frame_time = time_cs
        self.frame_time = time_cs
        self.frame_notes = 0
        self.frame_actions = 0
        if self.logical_time != time_cs:
            self.logical_time = time_cs
            self.logical_signatures.clear()
            self.logical_press_lanes.clear()
            self.logical_touch_areas.clear()

    def lane_blocked(self, lane: int) -> bool:
        assert self.frame_time is not None
        hold_end = self.active_holds.get(lane)
        if hold_end is not None and self.frame_time <= hold_end + RELEASE_COOLDOWN_CS:
            return True
        last_press = self.last_press_time.get(lane)
        return last_press is not None and self.frame_time - last_press < MIN_PRESS_GAP_CS

    def touch_blocked(self, area: int) -> bool:
        assert self.frame_time is not None
        hold_end = self.active_touch_holds.get(area)
        return hold_end is not None and self.frame_time <= hold_end + RELEASE_COOLDOWN_CS

    def add_note(self, note: _NoteRecord, action: bool) -> None:
        assert self.frame_time is not None
        if note.signature in self.logical_signatures:
            raise _InvalidPrefix
        if note.press_lane is not None:
            if note.press_lane in self.logical_press_lanes or self.lane_blocked(note.press_lane):
                raise _InvalidPrefix
            self.logical_press_lanes.add(note.press_lane)
            self.last_press_time[note.press_lane] = self.frame_time
        if note.touch_area is not None:
            if note.touch_area in self.logical_touch_areas or self.touch_blocked(note.touch_area):
                raise _InvalidPrefix
            self.logical_touch_areas.add(note.touch_area)
        if note.hold is not None:
            lane, duration = note.hold
            self.active_holds[lane] = self.frame_time + duration
        if note.touch_hold is not None:
            area, duration = note.touch_hold
            self.active_touch_holds[area] = self.frame_time + duration
        self.logical_signatures.add(note.signature)
        self.frame_notes += 1
        self.frame_actions += int(action)


@dataclass(frozen=True)
class ConstraintViolation:
    rule: str
    time_sec: float
    detail: str


def validate_frames(frames: list["Frame"]) -> list[ConstraintViolation]:
    """审计整首解析结果；返回全部违规，调用方据此整曲排除。"""
    violations: list[ConstraintViolation] = []
    state = _ReplayState()
    previous_time = 0
    for frame in frames:
        time_cs = round(frame.time_sec * 100)
        if time_cs < previous_time:
            violations.append(ConstraintViolation("FRAME_TIME_BACKWARD", frame.time_sec, f"{time_cs} < {previous_time}"))
            continue
        previous_time = time_cs
        frame_tokens = [SOS, *encode_frame(frame, time_cs / 100.0), EOS]
        frame_valid = True
        for pos, target in enumerate(frame_tokens):
            allowed = allowed_tokens(frame_tokens[:pos])
            if target not in allowed:
                violations.append(ConstraintViolation(
                    "FRAME_TOKEN_INVALID",
                    frame.time_sec,
                    f"位置={pos} token={target} allowed={allowed[:16]}",
                ))
                frame_valid = False
                break

        if not frame_valid:
            continue
        try:
            state.begin_frame(time_cs)
            _add_frame_records(state, frame)
        except _InvalidPrefix:
            violations.append(ConstraintViolation("RESOURCE_CONFLICT", frame.time_sec, "同位置 press、持续占用或冷却冲突"))
    return violations


def _add_frame_records(state: _ReplayState, frame: "Frame") -> None:
    from maidata_parser import NoteType

    for note in frame.notes:
        tokens = tuple(encode_note(note))
        if note.type is NoteType.TAP:
            lane = LANE_BASE + note.data.value - 1
            state.add_note(_NoteRecord(tokens, press_lane=lane), action=True)
        elif note.type is NoteType.HOLD:
            lane = LANE_BASE + note.data.lane.value - 1
            duration = round(note.data.holdTime * 100)
            if not 1 <= duration <= 2999:
                raise _InvalidPrefix
            state.add_note(_NoteRecord(tokens, press_lane=lane, hold=(lane, duration)), action=True)
        elif note.type is NoteType.TOUCH:
            area = TOUCH_BASE + note.data.Touch_area.value - 1
            state.add_note(_NoteRecord(tokens, touch_area=area), action=False)
        elif note.type is NoteType.TOUCH_HOLD:
            area = TOUCH_BASE + note.data.Touch_area.value - 1
            duration = round(note.data.holdTime * 100)
            if not 1 <= duration <= 2999:
                raise _InvalidPrefix
            state.add_note(_NoteRecord(tokens, touch_area=area, touch_hold=(area, duration)), action=False)
        else:
            if not note.data:
                raise _InvalidPrefix
            first = note.data[0]
            lane = LANE_BASE + first.start_lane.value - 1
            press_lane = None if first.isSlideNoHead else lane
            for segment in note.data:
                wait = round(segment.wait_duration * 100)
                trace = round(segment.trace_duration * 100)
                if not 0 <= wait <= 2999 or not 0 <= trace <= 2999 or wait == trace == 0:
                    raise _InvalidPrefix
            state.add_note(_NoteRecord(tokens, press_lane=press_lane), action=press_lane is not None)


def allowed_tokens(
    tokens: list[int],
    max_notes_per_frame: int = 33,
    max_action_notes_per_frame: int = 2,
    min_frame_time: int = 0,
    max_frame_time: int = 2999,
) -> tuple[int, ...]:
    """重放完整前缀，返回符合严格普通 Master 规范的下一批 token。"""
    if not tokens:
        return (SOS,)
    try:
        return _PrefixReplay(
            tokens,
            max_notes_per_frame=max_notes_per_frame,
            max_action_notes_per_frame=max_action_notes_per_frame,
            min_frame_time=min_frame_time,
            max_frame_time=max_frame_time,
        ).run()
    except _NeedToken as need:
        return need.allowed
    except _InvalidPrefix:
        return ()


class _PrefixReplay:
    def __init__(self, tokens: list[int], *, max_notes_per_frame: int, max_action_notes_per_frame: int,
                 min_frame_time: int, max_frame_time: int):
        self.tokens = tokens
        self.i = 0
        self.max_notes = max_notes_per_frame
        self.max_actions = max_action_notes_per_frame
        self.min_time = max(0, min_frame_time)
        self.max_time = min(2999, max_frame_time)
        self.state = _ReplayState()

    def run(self) -> tuple[int, ...]:
        self._take_exact(SOS)
        if self._at_end():
            raise _NeedToken(self._sequence_boundary_tokens())
        while True:
            token = self._peek()
            if token == EOS:
                self.i += 1
                if not self._at_end():
                    raise _InvalidPrefix
                return ()
            if token != FRAME_START:
                raise _InvalidPrefix
            self.i += 1
            self._parse_frame()
            if self._at_end():
                raise _NeedToken(self._sequence_boundary_tokens())

    def _parse_frame(self) -> None:
        min_time = max(self.state.previous_frame_time, self.min_time)
        if self._at_end():
            raise _NeedToken(tuple(range(TS_BASE + min_time, TS_BASE + self.max_time + 1)))
        time_token = self._take_from(TIMES)
        time_cs = time_token - TS_BASE
        if time_cs > self.max_time or time_cs < self.state.previous_frame_time:
            raise _InvalidPrefix
        self.state.begin_frame(time_cs)
        self._parse_note()
        while True:
            if self._at_end():
                raise _NeedToken(self._note_boundary_tokens())
            if self._peek() == FRAME_END:
                self.i += 1
                return
            self._parse_note()

    def _parse_note(self) -> None:
        allowed_types = self._note_types()
        note_type = self._take_from(allowed_types)
        if note_type == NOTE_TAP:
            note = self._parse_tap()
            self.state.add_note(note, action=True)
        elif note_type == NOTE_HOLD:
            note = self._parse_hold()
            self.state.add_note(note, action=True)
        elif note_type == NOTE_TOUCH:
            note = self._parse_touch()
            self.state.add_note(note, action=False)
        else:
            note = self._parse_slide()
            self.state.add_note(note, action=note.press_lane is not None)

    def _parse_tap(self) -> _NoteRecord:
        lane = self._take_from(self._available_press_lanes())
        attrs, remaining = self._parse_note_attrs()
        note = _NoteRecord((NOTE_TAP, lane, *attrs), press_lane=lane)
        if self._at_end():
            self._finish_note_options(note, action=True, extras=remaining)
        return note

    def _parse_hold(self) -> _NoteRecord:
        lane = self._take_from(self._available_press_lanes())
        duration = self._take_from(POSITIVE_TIMES) - TS_BASE
        attrs, remaining = self._parse_note_attrs()
        note = _NoteRecord((NOTE_HOLD, lane, duration, *attrs), press_lane=lane, hold=(lane, duration))
        if self._at_end():
            self._finish_note_options(note, action=True, extras=remaining)
        return note

    def _parse_touch(self) -> _NoteRecord:
        areas = tuple(area for area in TOUCHES if not self._touch_unavailable(area))
        area = self._take_from(areas)
        duration = None
        firework = False
        if self._at_end():
            note = _NoteRecord((NOTE_TOUCH, area, None, False), touch_area=area)
            self._finish_note_options(note, action=False, extras=POSITIVE_TIMES + (IS_FIREWORK,))
        if self._peek() in TIMES:
            duration = self._take_from(POSITIVE_TIMES) - TS_BASE
            if self._at_end():
                note = _NoteRecord((NOTE_TOUCH, area, duration, False), touch_hold=(area, duration), touch_area=area)
                self._finish_note_options(note, action=False, extras=(IS_FIREWORK,))
        if self._peek() == IS_FIREWORK:
            self.i += 1
            firework = True
            if self._at_end():
                hold = (area, duration) if duration is not None else None
                note = _NoteRecord((NOTE_TOUCH, area, duration, True), touch_hold=hold, touch_area=area)
                self._finish_note_options(note, action=False, extras=())
        signature = (NOTE_TOUCH, area, duration, firework)
        hold = (area, duration) if duration is not None else None
        return _NoteRecord(signature, touch_hold=hold, touch_area=area)

    def _parse_slide(self) -> _NoteRecord:
        note_attrs, remaining_attrs = self._parse_note_attrs()
        if self._at_end():
            raise _NeedToken(remaining_attrs + (SEGMENT_START,))
        if self._peek() != SEGMENT_START:
            raise _InvalidPrefix
        segments = []
        first_start = previous_end = None
        first_headed = True
        while True:
            self._take_exact(SEGMENT_START)
            segment, no_head = self._parse_slide_segment(
                first_start=first_start,
                previous_end=previous_end,
                first_segment=not segments,
            )
            segments.append(segment)
            if first_start is None:
                first_start = segment[1]
                first_headed = not no_head
            previous_end = segment[2]
            if self._at_end():
                signature = (NOTE_SLIDE, *note_attrs, tuple(segments))
                if signature in self.state.logical_signatures:
                    raise _NeedToken((SEGMENT_START,))
                note = _NoteRecord(signature, press_lane=first_start if first_headed else None)
                self._finish_note_options(note, action=first_headed, extras=(SEGMENT_START,))
            if self._peek() != SEGMENT_START:
                break
        assert first_start is not None
        signature = (NOTE_SLIDE, *note_attrs, tuple(segments))
        return _NoteRecord(signature, press_lane=first_start if first_headed else None)

    def _parse_slide_segment(self, *, first_start: int | None, previous_end: int | None,
                             first_segment: bool) -> tuple[tuple, bool]:
        shape = self._take_from(SHAPES)
        starts = LANES if first_segment else tuple(dict.fromkeys((first_start, previous_end)))
        start = self._take_from(starts)
        end = self._take_from(self._slide_end_lanes(shape, start))
        middle = None
        if shape == GRAND_V:
            middle = self._take_from(tuple(lane for lane in LANES if lane not in (start, end)))
        wait = self._take_from(TIMES) - TS_BASE
        traces = POSITIVE_TIMES if wait == 0 else TIMES
        trace = self._take_from(traces) - TS_BASE

        attrs = []
        attr_order = []
        if shape == CIRCLE:
            attr_order.append((IS_CW, IS_CCW))
        if first_segment:
            attr_order.extend(((IS_FORCE_STAR,), (IS_FAKE_ROTATE,)))
        attr_order.append((IS_SLIDE_BREAK,))
        if first_segment:
            attr_order.append((IS_SLIDE_NO_HEAD,))

        no_head = False
        for index, group in enumerate(attr_order):
            if self._at_end():
                remaining = tuple(token for rest in attr_order[index:] for token in rest)
                if self._slide_head_must_be_no_head(start, first_segment, no_head):
                    if IS_SLIDE_NO_HEAD not in remaining:
                        raise _InvalidPrefix
                    raise _NeedToken(remaining)
                raise _NeedToken(remaining + (SEGMENT_END,))
            if self._peek() in group:
                token = self._peek()
                self.i += 1
                attrs.append(token)
                no_head |= token == IS_SLIDE_NO_HEAD

        if self._slide_head_must_be_no_head(start, first_segment, no_head):
            raise _InvalidPrefix
        self._take_exact(SEGMENT_END)
        return (shape, start, end, middle, wait, trace, tuple(attrs)), no_head

    def _parse_note_attrs(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        attrs = []
        if not self._at_end() and self._peek() == IS_BREAK:
            self.i += 1
            attrs.append(IS_BREAK)
        if not self._at_end() and self._peek() == IS_EX:
            self.i += 1
            attrs.append(IS_EX)
        remaining = () if IS_EX in attrs else ((IS_EX,) if IS_BREAK in attrs else (IS_BREAK, IS_EX))
        return tuple(attrs), remaining

    def _finish_note_options(self, note: _NoteRecord, *, action: bool, extras: tuple[int, ...]) -> None:
        self.state.add_note(note, action=action)
        raise _NeedToken(extras + self._note_boundary_tokens())

    def _slide_head_must_be_no_head(self, start: int, first_segment: bool, no_head: bool) -> bool:
        return first_segment and not no_head and (
            len(self.state.logical_press_lanes) >= self.max_actions
            or start in self.state.logical_press_lanes
            or self.state.lane_blocked(start)
        )

    def _slide_end_lanes(self, shape: int, start: int) -> tuple[int, ...]:
        start_index = start - LANE_BASE
        if shape == LINE:
            deltas = (2, 3, 4, 5, 6)
        elif shape in (S_SHAPE, Z_SHAPE, WIFI):
            deltas = (4,)
        elif shape in (V_SHAPE, GRAND_V):
            deltas = (1, 2, 3, 4, 5, 6, 7)
        else:
            deltas = tuple(range(8))
        return tuple(LANE_BASE + (start_index + delta) % 8 for delta in deltas)

    def _available_press_lanes(self) -> tuple[int, ...]:
        return tuple(lane for lane in LANES if lane not in self.state.logical_press_lanes and not self.state.lane_blocked(lane))

    def _touch_unavailable(self, area: int) -> bool:
        return area in self.state.logical_touch_areas or self.state.touch_blocked(area)

    def _note_types(self) -> tuple[int, ...]:
        if self.state.frame_notes >= self.max_notes:
            return ()
        types = []
        if any(not self._touch_unavailable(area) for area in TOUCHES):
            types.append(NOTE_TOUCH)
        if len(self.state.logical_press_lanes) < self.max_actions:
            if self._available_press_lanes():
                types.extend((NOTE_TAP, NOTE_HOLD))
        # 达到 press 上限后仍允许 Slide，但第一段只能以 no-head 闭合。
        types.append(NOTE_SLIDE)
        return tuple(types)

    def _note_boundary_tokens(self) -> tuple[int, ...]:
        allowed = list(self._note_types())
        if self.state.frame_notes:
            allowed.append(FRAME_END)
        return tuple(allowed)

    def _sequence_boundary_tokens(self) -> tuple[int, ...]:
        min_time = max(self.state.previous_frame_time, self.min_time)
        if self.max_notes >= 1 and min_time <= self.max_time:
            return (FRAME_START, EOS)
        return (EOS,)

    def _take_exact(self, expected: int) -> int:
        return self._take_from((expected,))

    def _take_from(self, allowed: tuple[int, ...]) -> int:
        if not allowed:
            raise _InvalidPrefix
        if self._at_end():
            raise _NeedToken(allowed)
        token = self.tokens[self.i]
        if token not in allowed:
            raise _InvalidPrefix
        self.i += 1
        return token

    def _peek(self) -> int:
        if self._at_end():
            raise _NeedToken(())
        return self.tokens[self.i]

    def _at_end(self) -> bool:
        return self.i == len(self.tokens)


def _self_check() -> None:
    t = lambda cs: TS_BASE + cs
    l = lambda lane: LANE_BASE + lane - 1
    a = lambda area: TOUCH_BASE + area - 1

    assert allowed_tokens([]) == (SOS,)
    assert allowed_tokens([SOS]) == (FRAME_START, EOS)
    assert allowed_tokens([SOS], min_frame_time=2200, max_frame_time=2199) == (EOS,)
    assert allowed_tokens([SOS], max_notes_per_frame=0) == (EOS,)
    assert allowed_tokens([NOTE_TAP]) == ()
    assert allowed_tokens([SOS, SOS]) == ()
    assert allowed_tokens([SOS, EOS, FRAME_START]) == ()
    assert allowed_tokens([SOS, FRAME_START, t(1200), NOTE_TAP, FRAME_END]) == ()

    tap = [SOS, FRAME_START, t(1200), NOTE_TAP, l(1)]
    assert IS_BREAK in allowed_tokens(tap)
    assert IS_EX in allowed_tokens(tap)
    assert NOTE_TAP in allowed_tokens(tap)
    assert allowed_tokens(tap + [IS_EX, IS_BREAK]) == ()

    hold = [SOS, FRAME_START, t(1200), NOTE_HOLD, l(1)]
    assert t(0) not in allowed_tokens(hold)
    assert t(1) in allowed_tokens(hold)

    touch = [SOS, FRAME_START, t(1200), NOTE_TOUCH, a(1)]
    assert t(0) not in allowed_tokens(touch)
    assert t(1) in allowed_tokens(touch)

    # HOLD 结束于 13.00s；同 lane 在 13.05s 仍处于冷却，13.06s 才允许。
    hold_frame = [SOS, FRAME_START, t(1200), NOTE_HOLD, l(1), t(100), FRAME_END]
    blocked = hold_frame + [FRAME_START, t(1305), NOTE_TAP]
    released = hold_frame + [FRAME_START, t(1306), NOTE_TAP]
    assert l(1) not in allowed_tokens(blocked)
    assert l(2) in allowed_tokens(blocked)
    assert l(1) in allowed_tokens(released)

    # 同 lane 独立 press 小于 30ms 时禁止，刚好 30ms 时允许。
    tap_frame = [SOS, FRAME_START, t(1200), NOTE_TAP, l(1), FRAME_END]
    assert l(1) not in allowed_tokens(tap_frame + [FRAME_START, t(1202), NOTE_TAP])
    assert l(1) in allowed_tokens(tap_frame + [FRAME_START, t(1203), NOTE_TAP])

    # 同一逻辑 TS 最多两个独立 press；第三个有头 press 禁止，TOUCH 和 no-head Slide 允许。
    double_press = [
        SOS, FRAME_START, t(1200),
        NOTE_TAP, l(1), NOTE_HOLD, l(2), t(10),
    ]
    double_allowed = allowed_tokens(double_press)
    assert NOTE_TAP not in double_allowed
    assert NOTE_HOLD not in double_allowed
    assert NOTE_TOUCH in double_allowed
    assert NOTE_SLIDE in double_allowed
    third_slide = double_press + [NOTE_SLIDE, SEGMENT_START, LINE, l(3), l(5), t(10), t(10)]
    assert IS_SLIDE_NO_HEAD in allowed_tokens(third_slide)
    assert SEGMENT_END not in allowed_tokens(third_slide)
    assert SEGMENT_END in allowed_tokens(third_slide + [IS_SLIDE_NO_HEAD])

    # 相同 TS 即同一逻辑帧，同 lane 重复 press 仍然冲突。
    same_time = tap_frame + [FRAME_START, t(1200), NOTE_HOLD]
    assert l(1) not in allowed_tokens(same_time)

    # HOLD 占用 lane 上允许 no-head Slide，但有头 Slide 不能闭合第一段。
    slide = hold_frame + [FRAME_START, t(1250), NOTE_SLIDE, SEGMENT_START, LINE, l(1), l(3), t(10), t(10)]
    assert IS_SLIDE_NO_HEAD in allowed_tokens(slide)
    assert SEGMENT_END not in allowed_tokens(slide)
    assert SEGMENT_END in allowed_tokens(slide + [IS_SLIDE_NO_HEAD])

    # Slide 几何和方向属性。
    slide_start = [SOS, FRAME_START, t(1200), NOTE_SLIDE, SEGMENT_START, LINE, l(1)]
    assert l(1) not in allowed_tokens(slide_start)
    assert l(2) not in allowed_tokens(slide_start)
    assert l(3) in allowed_tokens(slide_start)
    line_attrs = slide_start + [l(3), t(10), t(10)]
    assert IS_CW not in allowed_tokens(line_attrs)
    circle_attrs = [SOS, FRAME_START, t(1200), NOTE_SLIDE, SEGMENT_START, CIRCLE, l(1), l(1), t(10), t(10)]
    assert IS_CW in allowed_tokens(circle_attrs)
    assert IS_CCW in allowed_tokens(circle_attrs)

    # 后续 segment 只能从第一段起点或前段终点开始。
    first_segment = [
        SOS, FRAME_START, t(1200), NOTE_SLIDE,
        SEGMENT_START, LINE, l(1), l(3), t(10), t(10), SEGMENT_END,
        SEGMENT_START, LINE,
    ]
    assert set(allowed_tokens(first_segment)) == {l(1), l(3)}

    # 重复的局部 segment 可以继续扩展，但完整 Slide 不能以重复状态结束。
    slide_frame = [
        SOS, FRAME_START, t(1200), NOTE_SLIDE,
        SEGMENT_START, LINE, l(1), l(3), t(10), t(10), IS_SLIDE_NO_HEAD, SEGMENT_END,
    ]
    duplicate_slide = slide_frame + [
        NOTE_SLIDE, SEGMENT_START, LINE, l(1), l(3), t(10), t(10), IS_SLIDE_NO_HEAD,
    ]
    assert SEGMENT_END in allowed_tokens(duplicate_slide)
    duplicate_closed = duplicate_slide + [SEGMENT_END]
    assert allowed_tokens(duplicate_closed) == (SEGMENT_START,)
    extended_slide = duplicate_closed + [
        SEGMENT_START, LINE, l(3), l(5), t(10), t(10), SEGMENT_END,
    ]
    extended_allowed = allowed_tokens(extended_slide)
    assert SEGMENT_START in extended_allowed
    assert NOTE_TOUCH in extended_allowed
    assert FRAME_END in extended_allowed

    # 同区域 TOUCH_HOLD 的 50ms 冷却。
    touch_hold = [SOS, FRAME_START, t(1200), NOTE_TOUCH, a(1), t(100), FRAME_END]
    assert a(1) not in allowed_tokens(touch_hold + [FRAME_START, t(1305), NOTE_TOUCH])
    assert a(1) in allowed_tokens(touch_hold + [FRAME_START, t(1306), NOTE_TOUCH])

    # 所有 Touch 区域都已在同一逻辑帧使用时，不能再暴露无法闭合的 NOTE_TOUCH。
    full_touch_frame = [SOS, FRAME_START, t(1200)]
    for area in TOUCHES:
        full_touch_frame.extend((NOTE_TOUCH, area))
    full_touch_allowed = allowed_tokens(full_touch_frame)
    assert NOTE_TOUCH not in full_touch_allowed
    assert FRAME_END in full_touch_allowed
    assert IS_FIREWORK in full_touch_allowed

    print("[constrained-decode] 自检通过")


if __name__ == "__main__":
    _self_check()
