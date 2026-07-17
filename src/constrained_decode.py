"""maimai token 严格解码规范（仅保留历史规则说明，不含实现）。

本文件曾用于旧自回归 token 模型。当前工程已经改为滑窗 BERT Encoder 等长预测，不再生成
token，因此严格前缀解码器、tokenizer 和整曲 token 审计实现均已删除。以下内容保留
为历史规范，不能作为当前训练或推理存在约束执行的依据。

旧 token 序列结构
==================

完整序列为 ``SOS (FRAME)* EOS``。每帧结构为
``FRAME_START TS NOTE+ FRAME_END``，时间戳不得倒退，同一个量化时间允许出现多个
frame，并在资源检查中视作同一个逻辑帧。PAD 不得出现在有效序列中，重复 SOS、内部
EOS、缺字段和 FRAME_END 后的垃圾 token 都属于死路，不能靠追加 token 修复。

旧 Note 结构
=============

- TAP：``NOTE_TAP LANE [IS_BREAK] [IS_EX]``。
- HOLD：``NOTE_HOLD LANE TS(duration) [IS_BREAK] [IS_EX]``，duration 必须为
  ``TS_1..TS_2999``。
- TOUCH：``NOTE_TOUCH AREA [TS(duration)] [IS_FIREWORK]``；省略 duration 表示普通
  TOUCH，正 duration 表示 TOUCH_HOLD。
- SLIDE：``NOTE_SLIDE [IS_BREAK] [IS_EX] SEGMENT+``。
- SEGMENT：``SEGMENT_START SHAPE START END [MIDDLE] WAIT TRACE [ATTRS]
  SEGMENT_END``。GrandV 必须有不同于起终点的 middle lane；Circle 才允许 CW/CCW；
  wait 和 trace 不能同时为零；头部属性只允许出现在第一段。

旧 Slide 几何和连接规则
=======================

- Line 只允许终点相对起点偏移 2..6。
- S、Z、Wifi 只允许偏移 4。
- V、GrandV 不允许同起终点。
- Circle、P、Q、PP、QQ 允许同起终点。
- 后续段起点只能是第一段起点或前一段终点，以支持分叉和链式 Slide。
- 相同局部 segment 可以闭合并继续扩展；只有完整 Slide Note 与同逻辑帧已有 Note
  重复时，旧实现才强制继续生成下一段。

旧资源规则
==========

- 每个物理 frame 最多 33 个 Note；同一逻辑时间最多两个独立 press。
- 独立 press 是 TAP、HOLD_START 和有头 SLIDE_HEAD；TOUCH、TOUCH_HOLD、no-head
  Slide 以及 Slide 后续段不计入。
- 同一逻辑时间、同一 lane 不能重复独立 press；同一 Touch area 不能重复 TOUCH 或
  TOUCH_HOLD，也不能同时出现两者。
- HOLD 占用对应 lane，TOUCH_HOLD 占用具体 area。占用结束后 50ms 内仍禁止同位置
  新输入，只有超过 50ms 才释放。
- 同 lane 相邻独立 press 间隔小于 30ms 时禁止，刚好 30ms 允许。
- HOLD 占用或 press 已满时，有头 Slide 禁止，但 no-head Slide 仍允许。
- 不同 lane 的输入、不同 Touch area、HOLD 期间的其他位置和 Slide 活跃期间的其他
  音符不能被粗粒度禁止。

旧实现实际覆盖
==============

删除前的实现包含：严格前缀重放、canonical 字段顺序、时间非递减、同 TS 逻辑资源
合并、frame Note/press 上限、Slide 几何与连接、同帧资源冲突、HOLD 和 TOUCH_HOLD
跨帧占用、50ms 冷却、30ms press 间隔、no-head 例外、重复完整 Slide 检查、候选
可闭合性和 malformed prefix 死路处理。

旧实现与早期顶部说明不完全一致的地方
========================================

- 旧文件曾写“尚未实现整曲审计”，但后来已经存在 ``validate_frames()``。它会返回
  全部发现的违规，并支持超时；不过它不是完整的数据清洗系统。
- ``validate_frames()`` 对每个 frame 单独重放 token 语法，再用简化的对象记录检查
  跨帧资源。它没有用一条完整 token 前缀证明与逐 token 状态机完全等价。
- 审计违规记录只有规则名、时间和简短详情，没有规范要求的歌曲路径、level、lane、
  area、前后事件和完整 Note 信息。
- 没有实现特殊谱面识别、Master 元数据异常检查、稳定 round-trip 检查、整库扫描统计、
  旋转不变性验证、训练/验证划分联动或自动整曲排除报告。
- 质量审计候选没有实现，包括 30ms 到 50ms press、极短或超长 HOLD、高密度 jack、
  异常 Slide 路线和 Slide 端点附近输入。
- 顶部规范要求“报告全部原因”，但单个 frame 的 token 检查遇到第一个非法 token 后
  就停止该 frame，无法列出同一 frame 内后续全部语法问题。
- 旧 ``allowed_tokens()`` 的 frame 上限参数按当前物理 frame 计数，而同 TS 拆成多个
  frame 时 Note 总数不会跨 frame 累加；同 TS 的 press 和签名会跨 frame 合并。
- 旧对象审计会量化持续时间并拒绝超范围值，但 token 编码器本身曾对时间做 clamp，
  因此“任何入口都绝不 clamp”并没有在整个旧系统中统一成立。

当前状态
========

当前工程没有 token 词表、tokenizer、严格解码状态机或运行时规则校验。BERT 输出只经过
短音间隔、持续区间最短长度和固定 lane 映射等后处理。若未来恢复自回归 token 模型，
必须重新实现并测试本规范，不能直接把本文件当作已执行功能。
"""
