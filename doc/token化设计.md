# maimai Token 设计文档

## 一、词表总览

词表大小：**3072**

| 分类 | Token 名称 | ID 范围 | 数量 |
|------|-----------|---------|------|
| 基础控制符 | `<PAD>`, `<SOS>`, `<EOS>` | 0 ~ 2 | 3 |
| 帧结构分隔符 | `<FRAME_START>`, `<FRAME_END>` | 3 ~ 4 | 2 |
| 时间 | `TS_0` ~ `TS_2999`（兼作时间戳与 duration，单位 10ms） | 5 ~ 3004 | 3000 |
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

simai 的 `/` EACH 和反引号伪 EACH 都会在解析阶段合并到同一个 `Frame`。当前训练不保留伪 EACH 原始的 1ms 先后差异，因为 TS token 的精度是 10ms：

```text
1/2       -> 同一 Frame 内的 TAP 1 + TAP 2
1`2       -> 同一 Frame 内的 TAP 1 + TAP 2
1`2`3/4   -> 同一 Frame 内的 TAP 1 + TAP 2 + TAP 3 + TAP 4
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

- `TS_<持续帧数>`：持续时间，单位为 10ms
- 无时长括号的伪 HOLD（如 `5h`、`5bh`、`5xh`）在解析阶段归一化为对应的 TAP，不生成 `NOTE_HOLD` 或 `TS_0`。

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
- 无时长括号的伪 TOUCH HOLD（如 `Ch`、`Chf`）在解析阶段归一化为普通 TOUCH。

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

仅 `SLIDE_SHAPE_4`（GrandV）需要 `LANE_<拐点>`，放在 `LANE_<终点>` 之后：

```
SLIDE_SHAPE_4 LANE_起点 LANE_终点 LANE_拐点 TS_wait TS_trace
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

simai 中以 `*` 分隔、共享起点的多重 Slide，会编码为一个 `NOTE_SLIDE` 下的多个 segment。`*` 后的 segment 起点重置为原始起点：

```
<FRAME_START> TS_300
  NOTE_SLIDE
    <SEGMENT_START> SLIDE_SHAPE_1 LANE_1 LANE_3 TS_24 TS_48 <SEGMENT_END>
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

紧凑 EACH 也会展开为同一 Frame 内的多个 TAP，包括各 Tap 自己的属性，例如 `1b5b`、`2x6x`。

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
      SLIDE_SHAPE_4 LANE_1 LANE_5 LANE_3
      TS_24 TS_48
      IS_CW IS_SLIDE_BREAK
    <SEGMENT_END>
<FRAME_END>
```

对应原始文本：`1V35[8:3]`（顺时针 GrandV，起点 1，拐点 3，终点 5）

### 示例 5：无头 + 强制星星头的链式 Slide

```
<FRAME_START> TS_450
  NOTE_SLIDE
    <SEGMENT_START>
      SLIDE_SHAPE_1 LANE_1 LANE_5 TS_12 TS_24 IS_FORCE_STAR IS_SLIDE_NO_HEAD
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
    <SEGMENT_START> SLIDE_SHAPE_7 LANE_1 LANE_4 TS_24 TS_48 IS_SLIDE_BREAK <SEGMENT_END>
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

解析器不会返回 `NoteType.SLIDE(data=[])`。任何无法解析出完整 segment 的源 token 都会抛错，缓存构建随即终止，避免仅含 `NOTE_SLIDE` 的坏标签进入训练索引。

### 4.5 音符级属性与段落级属性不冲突

`IS_BREAK`（Note 级）和 `IS_SLIDE_BREAK`（段落级）可同时存在：
- `IS_BREAK`：影响评分系统
- `IS_SLIDE_BREAK`：影响该段视觉效果

原始文本 `1-7-2-6-3-5b[4:10]` 中，`b` 使整条 Note 的 `IS_BREAK=True`，同时末段的 `IS_SLIDE_BREAK=True`。

### 4.6 TOUCH_HOLD 与 TOUCH 的区分

通过持续帧数区分：
- `NOTE_TOUCH TOUCH_<区域>` 或 `NOTE_TOUCH TOUCH_<区域> TS_0` → 普通 TOUCH
- `NOTE_TOUCH TOUCH_<区域> TS_<非零>` → TOUCH_HOLD

### 4.7 simai 解析归一化与错误处理

以下是预期归一化，记录普通信息日志，不视为警告：

- 反引号伪 EACH 合并为普通 EACH。
- 伪 HOLD 转为 TAP，伪 TOUCH HOLD 转为 TOUCH。
- 独立 `$` 星形 TAP 转为普通 TAP；当前词表不保留 TAP 的星形外观。

以下表示源谱存在可明确修复的脏数据，解析器会输出警告并采用窄范围容错：

- EACH 出现空分支，例如 `/1/3`。
- TAP 后出现没有 HOLD/Slide 语义的孤立时长，例如 `2[8:1]`。
- 时值定义后多出 `}`，例如 `{2}}8h[64:47]`。
- 链式 Slide 后续段漏写形状符时，复用上一段形状；当前真实样例为 `8?-58[...]`，按 `8?-5-8[...]` 处理。

其他未知字符、无效轨道/Touch 区域、不完整 Slide、空 Slide 或无法完整消费的 token 都会抛出带歌曲、难度、槽位、时间和原始 token 的错误。`chart_cache` 不再跳过解析失败的歌曲，而是终止缓存构建。

Slide parser 支持真实谱面中属性出现在不同位置的写法，例如：

```text
2pbx7b[8:3]*p8b[8:3]
```

该 token 编码为一个 `NOTE_SLIDE`、两个完整 segment；`b/x` 会分别归一化为已有的 Break/EX 属性，不会产生空 Slide。

---

## 五、已知限制

### 5.1 时间精度

`TS_0`~`TS_2999` 以 10ms 为固定单位表示 `0.00s`~`29.99s`，与 mel 帧率无关。正的 duration 若四舍五入为 0，会提升为 `TS_1`。

### 5.2 链式 Slide 的 trace 分配

原始文本 `1-7-2-6-3-5b[4:10]` 中，`[4:10]` 是整条链的总时长。当前 parser 将全部 trace 分配给末段，中间段 trace=0。

token 格式可以表达更精确的语义（每段分配独立时长），但需要修改 parser 的分配逻辑。当前以 parser 行为准。

### 5.3 Wifi 形状的终点

Wifi（`w`）在游戏中有 3 个终点，但 token 格式只存 1 个终点（对侧端点，距离恒为 4）。另外 2 个终点由 `起点 ± 2 mod 8` 推导，无需显式存储。

### 5.4 多重 Slide 的分支属性还原尚未完整实现

当前已经实现多重 Slide 的基本往返：

- parser 遇到 `*` 时，会将下一段起点重置为第一个 segment 的起点。
- 一个多重 Slide 编码为一个 `NOTE_SLIDE` 下的多个 segment。
- token 解码后，若后续 segment 的起点等于首段起点且不等于前一段终点，生成器会重新输出 `*`。
- 各分支的形状、起点、终点、GrandV 拐点、等待时间和滑动时间可以进入 token 并还原。

以下部分尚未完整实现，因此目前不能保证多重 Slide 严格无损往返：

1. `*` 后各分支自己的 `?`、`!`、`@`、`$`、`$$` 尚未作为严格独立的分支状态处理。解析器内部部分状态会沿用到后续 segment，生成器又只在第一个 segment 前输出这些修饰符。
2. `x` 当前存储在 `Note.isEx`，不是 segment 属性，因此无法区分它原本写在首分支还是某个 `*` 分支。
3. `b` 的原始书写位置不能无损保留。解析器会将多种位置的 `b` 归一化为现有 Note/segment Break 属性；生成器目前只根据最后一个 segment 在整个 Slide 末尾输出 `b`。
4. `@`、`?`、`!` 都归入 `IS_SLIDE_NO_HEAD`，token 中无法区分原始写法，生成时统一使用 `?`。
5. parser 只保存语义化后的 segment 列表，不保存原始文本布局，因此即使语义相同，生成结果也不保证与输入字符串逐字符一致。

例如：

```text
1-4[8:1]b*-6[8:1]b
```

当前可以解析为共享起点的两个完整 segment，也不会生成空 Slide；但重新生成时不保证两个分支各自的 `b` 仍出现在原位置。只有不依赖上述分支级修饰符差异的普通多重 Slide，才能认为基本结构往返正确。

---

## 六、存储格式

### 6.1 `to_tensor`返回值要求

```python
-> list[absolute_time_offset:float], list[torch.Tensor]

assert len(list[absolute_time_offset]) == len(list[torch.Tensor])
```

| 维度 | 含义 | 使用者 |
|------|------|--------|
| `absolute_time_offset` | 绝对时间偏移（用于和音频对齐） | data loader |
| `token_sequence` | 一维 token 序列 | 模型输入 |

每一段`token_sequence`，的起始需要是`SOS`，终止需要为`EOS`

### 6.2 动态分段策略

**核心约束**：TS_0 ~ TS_2999 以 10ms 为单位，最多表示到 29.99 秒。每段内出现的相对时间必须落在这个范围内。

**不按固定 30 秒硬切**，而是在 token 累积接近 30 秒时，在最后一个完整 token 的结束处于下一个的起始的中间，在30s前切割。这样不会切碎任何音符/帧。

**绝对不能只看当前帧的 end**，必须看全局活跃音符的最晚 end（也就是从长按按下去到长按结束这一段时间，会有其他 token。这些其他的token也需要被正确处理）

```
音频: |───────────────────────────────────────────────────|
      0s                                            163.9s

硬切 30s:  |──30s──|──30s──|──30s──|──30s──|──30s──|─14s─|
           可能切在帧中间，破坏完整性

动态切:    |─28.5s─|─29.2s─|─28.8s─|─29.7s─|─29.4s─|─28.5s─|─19.8s─|
           每段 ≤30s，在 token 边界处切割，不破坏任何音符
```

**关键**：先判断再加入，确保每段 ≤ 30s。

#### 分段示例

假设一首 163.9 秒的歌曲，TS 仍使用固定 10ms 精度：

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

> 当前实现补充：历史设计使用 `to_tensor()` 动态分段；当前训练代码已改成滑窗训练。下面的动态分段流程仍描述 token 格式的原始设计约束，不等同于当前 `ChartDataset` 的训练样本构造方式。

第一步：重建缓存。

第二步：验证数据集。
需满足以下两个条件，才能将它们确认为一个数据集：
(a) 验证有缓存的音频文件与对应的谱面文件能否对应；
(b) 验证谱面文件在解析时没有报错。

解析失败属于缓存构建错误，不能静默跳过歌曲。解析语义变化时必须提升 `src/chart_cache.py` 的 `CACHE_VERSION`；当前版本为 `4`。

第三步：编译谱面并计算总长度。
每一个谱面都需要经过编译。编译后可以得到有多少个段，每一段代表一个长度。要把所有有可能的段全部编译一遍，算出全部的长度，加在一起就是总的长度，并且准备好索引。

第四步：迭代读取与音频切片。
只有在迭代器运行的时候，从缓存中读取数据，算出正确的偏移切片。动态分段设计里每段 token 的相对时间必须 ≤30 秒，所以音频切片也应覆盖对应的 token 时间范围。
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

### 6.3.1 当前滑窗训练实现

当前 `src/dataset.py` 不再直接使用 `to_tensor()` 的动态分段作为训练样本，而是按固定提交区滑动构造样本：

```text
target_start = 0s, 1s, 2s, ...（训练）
target_start = 0s, 10s, 20s, ...（验证和正式推理提交）
逻辑窗口起点 = target_start - 12s
token 前缀区 = 相对 6..12s
提交区 = 相对 12..22s
目标 token = 提交区内“开始”的音符
跨边界 HOLD/SLIDE = 不拆开，归属于开始时间所在提交区
```

输入音频窗口由模型结构决定。当前模型配置是：

```text
sample_rate = 22050
hop_length = 256
n_audio_ctx = 1500
mel_frames = 2 * n_audio_ctx = 3000
输入窗口时长 = 3000 / (22050 / 256) ≈ 34.83s
```

因此当前实现里的“输入窗口”不是严格 30 秒，而是 **3000 个 mel 帧，约 34.83 秒**。

这不违反 `TS_0 ~ TS_2999` 的 30 秒约束，原因是：

```text
TS 的 30 秒约束限制的是 token 序列里的相对时间可表达范围
当前训练目标只覆盖相对 12..22 秒的 10 秒提交区
输入 mel 可以包含额外上下文，只要目标 token 的相对时间仍在 TS_0~TS_2999 内
```

当前默认上下文关系是：

```text
输入窗口: 约 34.83s
左音频上下文: 12.0s
token 前缀区: 6.0s（相对 6..12s，只作为 decoder 上下文）
提交区: 10.0s（相对 12..22s）
右音频上下文: 约 12.83s
```

如果以后要让音频输入窗口也严格等于 30 秒，需要同步修改：

```text
src/train.py 的 n_audio_ctx
src/dataset.py 的 mel_frames
src/infer.py 的 WINDOW_FRAMES
src/overfit_window.py / src/overfit_song.py 的模型尺寸和 collate mel_frames
所有旧 checkpoint 都不能继续使用
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

当前 decoder 没有传入 token attention mask。PAD 只出现在序列尾部，因果 attention 不会影响此前有效 token；训练损失由独立 `loss_mask` 限定，只监督目标区 token 和 EOS，不监督 PAD 或前缀。

### 6.5 与音频对齐

每个 token 的时间位置计算：

```
token_absolute_time = segment_offset + token_relative_time
```

其中 `token_relative_time` 从 token 序列中的 `TS_xxx` 解析。data loader 用这个绝对时间从音频特征序列中取对应的帧。
