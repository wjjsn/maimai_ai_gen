# maimai Token 设计文档

## 一、词表总览

词表大小：**3072**

| 分类 | Token 名称 | ID 范围 | 数量 |
|------|-----------|---------|------|
| 基础控制符 | `<PAD>`, `<SOS>`, `<EOS>` | 0 ~ 2 | 3 |
| 帧结构分隔符 | `<FRAME_START>`, `<FRAME_END>` | 3 ~ 4 | 2 |
| 时间/帧数 | `TS_0` ~ `TS_2999`（兼作时间戳与 duration 帧数） | 5 ~ 3004 | 3000 |
| 轨道位置 | `LANE_1` ~ `LANE_8`（Tap 轨、Slide 起点/终点/拐点） | 3005 ~ 3012 | 8 |
| 触控区域 | `TOUCH_1` ~ `TOUCH_33`（A1~A8, B1~B8, C, D1~D8, E1~E8） | 3013 ~ 3045 | 33 |
| 音符类型 | `NOTE_TAP`, `NOTE_TOUCH`, `NOTE_HOLD`, `NOTE_SLIDE` | 3046 ~ 3049 | 4 |
| Slide 段落包围符 | `<SEGMENT_START>`, `<SEGMENT_END>` | 3050 ~ 3051 | 2 |
| 滑条形状 | `SLIDE_SHAPE_1` ~ `SLIDE_SHAPE_11` | 3052 ~ 3062 | 11 |
| 音符级属性 | `IS_BREAK`, `IS_EX`, `IS_FIREWORK` | 3063 ~ 3065 | 3 |
| Slide 段落级方向 | `IS_CW`（顺时针 `>`）, `IS_CCW`（逆时针 `<`） | 3066 ~ 3067 | 2 |
| Slide 段落级标记 | `IS_FORCE_STAR`, `IS_FAKE_ROTATE`, `IS_SLIDE_BREAK`, `IS_SLIDE_NO_HEAD` | 3068 ~ 3071 | 4 |

---

## 二、各类型编码规范

### 2.1 帧结构

每一帧用 `<FRAME_START>` 开头、`<FRAME_END>` 结尾。帧内可包含多个音符。

```
<FRAME_START> TS_<时间戳帧数> <音符1> <音符2> ... <FRAME_END>
```

### 2.2 TAP

```
NOTE_TAP LANE_<n> [IS_BREAK] [IS_EX]
```

- `LANE_<n>`：轨道位置，n = 1~8
- `IS_BREAK`：Break Tap（可选）
- `IS_EX`：EX Tap（可选）

### 2.3 HOLD

```
NOTE_HOLD LANE_<n> TS_<持续帧数> [IS_BREAK] [IS_EX]
```

- `TS_<持续帧数>`：持续时间，用帧数表示

### 2.4 TOUCH

```
NOTE_TOUCH TOUCH_<区域> [IS_FIREWORK]
```

- `TOUCH_<区域>`：触控区域，1~33 对应 A1~A8, B1~B8, C, D1~D8, E1~E8
- `IS_FIREWORK`：烟花效果（可选）

### 2.5 TOUCH_HOLD

用 `NOTE_TOUCH` + 非零持续帧数 表达：

```
NOTE_TOUCH TOUCH_<区域> TS_<持续帧数> [IS_FIREWORK]
```

- 持续帧数 > 0 时为 TOUCH_HOLD
- 持续帧数 = 0 或省略时为普通 TOUCH

### 2.6 SLIDE

```
NOTE_SLIDE [IS_BREAK] [IS_EX]
  <SEGMENT_START>
    SLIDE_SHAPE_<形状编号>
    LANE_<起点> LANE_<终点> [LANE_<拐点>]
    TS_<等待帧数> TS_<划动帧数>
    [IS_CW | IS_CCW]
    [IS_FORCE_STAR] [IS_FAKE_ROTATE] [IS_SLIDE_BREAK] [IS_SLIDE_NO_HEAD]
  <SEGMENT_END>
  [<SEGMENT_START> ... <SEGMENT_END>] ...
```

#### 段落级属性说明

| 属性 | 含义 | 备注 |
|------|------|------|
| `IS_CW` | 顺时针弧线（`>`） | 仅 Circle 形状有效，与 IS_CCW 互斥 |
| `IS_CCW` | 逆时针弧线（`<`） | 仅 Circle 形状有效，与 IS_CW 互斥 |
| （两者都不出现） | 自动选最短路径（`^`） | Circle 形状的默认值 |
| `IS_FORCE_STAR` | 强制星星头（`$`） | |
| `IS_FAKE_ROTATE` | 假旋转（`$$`） | |
| `IS_SLIDE_BREAK` | 该段 Break（`b`） | |
| `IS_SLIDE_NO_HEAD` | 无头星星（`?` 或 `!` 或 `@`） | |

#### Note 级 vs 段落级

| 属性 | 级别 | 说明 |
|------|------|------|
| `IS_BREAK` | Note 级 | 整条 Slide 的 Break 属性，影响评分 |
| `IS_EX` | Note 级 | 整条 Slide 的 EX 属性 |
| `IS_CW` / `IS_CCW` | **段落级** | 每段独立，链式 slide 中不同段可有不同方向 |
| `IS_FORCE_STAR` | **段落级** | 每段独立 |
| `IS_FAKE_ROTATE` | **段落级** | 每段独立 |
| `IS_SLIDE_BREAK` | **段落级** | 每段独立，链式 slide 中可仅末段为 Break |
| `IS_SLIDE_NO_HEAD` | **段落级** | 每段独立 |

#### 形状编号映射

| 编号 | 形状 | simai 符号 | 说明 |
|------|------|-----------|------|
| 1 | Line | `-` | 直线 |
| 2 | Circle | `>` `<` `^` | 弧线（需配合 IS_CW/IS_CCW） |
| 3 | V | `v` | V 形折线 |
| 4 | GrandV | `V` | L 形折线（需 LANE_拐点） |
| 5 | P | `p` | 单弯 |
| 6 | Q | `q` | 单弯镜像 |
| 7 | PP | `pp` | 大弯 |
| 8 | QQ | `qq` | 大弯镜像 |
| 9 | S | `s` | 闪电形 |
| 10 | Z | `z` | 闪电镜像 |
| 11 | Wifi | `w` | 扇形 |

#### GrandV 的拐点

仅 `SLIDE_SHAPE_4`（GrandV）需要 `LANE_<拐点>`，插在 `LANE_<终点>` 之前：

```
SLIDE_SHAPE_4 LANE_起点 LANE_拐点 LANE_终点 TS_wait TS_trace
```

其他形状省略拐点：

```
SLIDE_SHAPE_<n> LANE_起点 LANE_终点 TS_wait TS_trace
```

#### 链式 Slide（Chaining）

多个 `<SEGMENT_START>...<SEGMENT_END>` 连续出现，前一段的终点 = 后一段的起点：

```
NOTE_SLIDE IS_BREAK
  <SEGMENT_START> SLIDE_SHAPE_1 LANE_1 LANE_7 TS_24 TS_0 <SEGMENT_END>
  <SEGMENT_START> SLIDE_SHAPE_1 LANE_7 LANE_2 TS_24 TS_0 <SEGMENT_END>
  <SEGMENT_START> SLIDE_SHAPE_1 LANE_2 LANE_6 TS_24 TS_0 <SEGMENT_END>
  <SEGMENT_START> SLIDE_SHAPE_1 LANE_6 LANE_3 TS_24 TS_0 <SEGMENT_END>
  <SEGMENT_START> SLIDE_SHAPE_1 LANE_3 LANE_5 TS_24 TS_512 IS_SLIDE_BREAK <SEGMENT_END>
```

对应原始文本：`1-7-2-6-3-5b[4:10]`

#### 多重 Slide（Multiple）

同一帧内多个 `NOTE_SLIDE`，各自独立：

```
<FRAME_START> TS_300
  NOTE_SLIDE
    <SEGMENT_START> SLIDE_SHAPE_1 LANE_1 LANE_3 TS_24 TS_48 <SEGMENT_END>
  NOTE_SLIDE
    <SEGMENT_START> SLIDE_SHAPE_1 LANE_1 LANE_4 TS_24 TS_48 IS_SLIDE_BREAK <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`1-3[8:1]*-4b[8:1]`

#### EACH 组合

同一帧内不同类型的音符并列：

```
<FRAME_START> TS_100
  NOTE_TAP LANE_1
  NOTE_TAP LANE_8
  NOTE_SLIDE
    <SEGMENT_START> SLIDE_SHAPE_1 LANE_3 LANE_7 TS_24 TS_24 <SEGMENT_END>
<FRAME_END>
```

---

## 三、完整示例

### 示例 1：Break EX Tap + 烟花 Touch

```
<FRAME_START> TS_100
  NOTE_TAP LANE_1 IS_BREAK IS_EX
  NOTE_TOUCH TOUCH_1 IS_FIREWORK
<FRAME_END>
```

### 示例 2：HOLD

```
<FRAME_START> TS_200
  NOTE_HOLD LANE_5 TS_48
<FRAME_END>
```

### 示例 3：TOUCH_HOLD

```
<FRAME_START> TS_200
  NOTE_TOUCH TOUCH_17 TS_96 IS_FIREWORK
<FRAME_END>
```

（TOUCH_17 = C 区域，持续 96 帧，带烟花）

### 示例 4：GrandV Slide（带拐点、顺时针、Break）

```
<FRAME_START> TS_300
  NOTE_SLIDE
    <SEGMENT_START>
      SLIDE_SHAPE_4 LANE_1 LANE_3 LANE_5
      TS_24 TS_48
      IS_CW IS_SLIDE_BREAK
    <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`1V35[8:3]`（顺时针 GrandV，起点 1，拐点 3，终点 5）

### 示例 5：无头 + 强制星星头的链式 Slide

```
<FRAME_START> TS_450
  NOTE_SLIDE IS_SLIDE_NO_HEAD IS_FORCE_STAR
    <SEGMENT_START>
      SLIDE_SHAPE_1 LANE_1 LANE_5 TS_12 TS_24
    <SEGMENT_END>
    <SEGMENT_START>
      SLIDE_SHAPE_2 LANE_5 LANE_8 TS_0 TS_36 IS_CCW IS_FAKE_ROTATE
    <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`1?-5[4:2]<8[4:3]$$`

### 示例 6：EACH（双押 Tap + Slide）

```
<FRAME_START> TS_600
  NOTE_TAP LANE_1
  NOTE_TAP LANE_8
  NOTE_SLIDE
    <SEGMENT_START>
      SLIDE_SHAPE_1 LANE_3 LANE_7 TS_24 TS_24
    <SEGMENT_END>
<FRAME_END>
```

### 示例 7：多重 Slide（同起点两条不同路线）

```
<FRAME_START> TS_700
  NOTE_SLIDE
    <SEGMENT_START> SLIDE_SHAPE_8 LANE_1 LANE_6 TS_24 TS_48 <SEGMENT_END>
  NOTE_SLIDE IS_SLIDE_BREAK
    <SEGMENT_START> SLIDE_SHAPE_7 LANE_1 LANE_4 TS_24 TS_48 <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`1qq6[8:3]*pp4b[8:3]`

### 示例 8：链式 Slide 中不同段有不同方向

```
<FRAME_START> TS_800
  NOTE_SLIDE
    <SEGMENT_START>
      SLIDE_SHAPE_1 LANE_6 LANE_2 TS_24 TS_0
    <SEGMENT_END>
    <SEGMENT_START>
      SLIDE_SHAPE_2 LANE_2 LANE_5 TS_24 TS_24 IS_CW
    <SEGMENT_END>
    <SEGMENT_START>
      SLIDE_SHAPE_1 LANE_5 LANE_8 TS_24 TS_0
    <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`6-2>5-8[8:3]`

- 第一段 Line：无方向（auto）
- 第二段 Circle：顺时针（CW）
- 第三段 Line：无方向（auto）

---

## 四、设计约束与边界情况

### 4.1 IS_CW / IS_CCW 仅 Circle 形状有效

对非 Circle 形状出现 IS_CW / IS_CCW 时应忽略。Circle 形状下两者都不出现 = 自动选最短路径（`^`）。

### 4.2 GrandV 的 LANE_拐点必选

`SLIDE_SHAPE_4` 必须有 `LANE_拐点`，其他形状不得出现。

### 4.3 TS_0 的含义

- 作为时间戳：表示第 0 帧
- 作为 duration：表示持续 0 帧（用于链式 slide 中间段的 trace）

### 4.4 NOTE_SLIDE 无 SEGMENT 的情况

`NOTE_SLIDE` 后必须跟至少一个 `<SEGMENT_START>...<SEGMENT_END>`。没有段落的 `NOTE_SLIDE` 为非法序列。

### 4.5 音符级属性与段落级属性不冲突

`IS_BREAK`（Note 级）和 `IS_SLIDE_BREAK`（段落级）可同时存在：
- `IS_BREAK`：影响评分系统
- `IS_SLIDE_BREAK`：影响该段视觉效果

原始文本 `1-7-2-6-3-5b[4:10]` 中，`b` 使整条 Note 的 `IS_BREAK=True`，同时末段的 `IS_SLIDE_BREAK=True`。

### 4.6 TOUCH_HOLD 与 TOUCH 的区分

通过持续帧数区分：
- `NOTE_TOUCH TOUCH_<区域>` 或 `NOTE_TOUCH TOUCH_<区域> TS_0` → 普通 TOUCH
- `NOTE_TOUCH TOUCH_<区域> TS_<非零>` → TOUCH_HOLD

---

## 五、已知限制

### 5.1 时间精度

`TS_0`~`TS_2999` 用帧数表示时间。需要约定帧率（如 100fps），否则无法还原绝对时间。

### 5.2 链式 Slide 的 trace 分配

原始文本 `1-7-2-6-3-5b[4:10]` 中，`[4:10]` 是整条链的总时长。当前 parser 将全部 trace 分配给末段，中间段 trace=0。

token 格式可以表达更精确的语义（每段分配独立时长），但需要修改 parser 的分配逻辑。当前以 parser 行为准。

### 5.3 Wifi 形状的终点

Wifi（`w`）在游戏中有 3 个终点，但 token 格式只存 1 个终点（对侧端点，距离恒为 4）。另外 2 个终点由 `起点 ± 2 mod 8` 推导，无需显式存储。

---

## 六、张量存储格式

### 6.1 三维张量结构

```python
shape = (num_segments, max_time_offset, token_sequence)
```

| 维度 | 含义 | 使用者 |
|------|------|--------|
| 第一维 `num_segments` | 分段数量（音频总时长 ÷ ~30s） | data loader |
| 第二维 `max_time_offset` | 段内时间偏移（用于和音频帧对齐） | data loader |
| 第三维 `token_sequence` | 一维 token 序列 | **模型输入** |

模型最终拿到的是第三维的一维 token 序列。第一维和第二维仅用于 data loader 的音频对齐和分段管理。

### 6.2 动态分段策略

**核心约束**：TS_0 ~ TS_2999 最多表示 30 秒（100fps × 3000 帧 = 30s）。每段必须 **≤ 30 秒**，不能超。

**不按固定 30 秒硬切**，而是在 token 累积接近 30 秒时，在最后一个完整 token 的结束处提前切割。这样不会切碎任何音符/帧。

```
音频: |───────────────────────────────────────────────────|
      0s                                            163.9s

硬切 30s:  |──30s──|──30s──|──30s──|──30s──|──30s──|─14s─|
           可能切在帧中间，破坏完整性

动态切:    |─28.5s─|─29.2s─|─28.8s─|─29.7s─|─29.4s─|─28.5s─|─19.8s─|
           每段 ≤30s，在 token 边界处切割，不破坏任何音符
```

#### 分段算法

```
offset = 0
tokens_accumulated = []
for token in full_token_sequence:
    if token 的结束时间 - offset > 30s:
        # 加入这个 token 会超 30s，先提交当前段
        提交当前段: (offset, tokens_accumulated)
        offset = token 的开始时间
        tokens_accumulated = []
    tokens_accumulated.append(token)
提交最后一段（如果还有剩余）
```

**关键**：先判断再加入，确保每段 ≤ 30s。

#### 分段示例

假设一首 163.9 秒的歌曲，100fps 帧率：

| 段号 | 偏移起点 | 偏移终点 | 实际时长 | token 范围 |
|------|---------|---------|---------|----------|
| 0 | 0.00s | 28.47s | 28.47s | TS_0 ~ TS_2846 |
| 1 | 28.47s | 57.69s | 29.22s | TS_0 ~ TS_2921 |
| 2 | 57.69s | 86.49s | 28.80s | TS_0 ~ TS_2879 |
| 3 | 86.49s | 116.19s | 29.70s | TS_0 ~ TS_2969 |
| 4 | 116.19s | 145.59s | 29.40s | TS_0 ~ TS_2939 |
| 5 | 145.59s | 163.90s | 18.31s | TS_0 ~ TS_1830 |

每段内部 TS 从 0 重新计数。data loader 记录每段的绝对偏移量，用于和音频特征对齐。

### 6.3 Data Loader 工作流程

```
1. 读取音频，计算总时长
2. 按动态分段策略将 token 序列切成 N 段
3. 每段记录:
    - segment_offset: 该段在音频中的绝对起始时间
    - tokens: 该段的 token 序列（一维）
    - time_slots: 每个 token 对应的相对时间（用于音频对齐）
4. 加载对应时间段的音频特征（mel spectrogram 等）
5. 将 token 序列和音频特征一起送入模型
```

### 6.4 张量填充

由于每段的 token 数量不等，需要填充到统一长度：

```python
# 每段 token 数量分布（实测数据）
# 中位数: ~1644 tokens/段
# 最大值: ~5490 tokens/段

padded_tensor = torch.full(
    (num_segments, max_tokens_per_segment),
    fill_value=PAD_TOKEN_ID,  # 0
    dtype=torch.long
)
```

模型通过 attention mask 忽略 PAD 位置。

### 6.5 与音频对齐

每个 token 的时间位置计算：

```
token_absolute_time = segment_offset + token_relative_time
```

其中 `token_relative_time` 从 token 序列中的 `TS_xxx` 解析。data loader 用这个绝对时间从音频特征序列中取对应的帧。
