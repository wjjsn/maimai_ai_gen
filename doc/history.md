# maimai 谱面生成模型历史记录

> 本文记录当时的排查过程、实验命令和已废弃实现。当前行为以 `AGENTS.md`、`src/` 和 `doc/token化设计.md` 为准。尤其不要沿用本文中的 `--mode overlap`、`--context-sec`、20 秒提交区、尾窗夹动或“每窗从 SOS 独立解码”等旧方案描述。

时间范围: 单会话内的排查 → 修复 → 单曲过拟合闭环 → 推理模式升级

## 一、初始现象

用户反馈: 训练出来的模型生成效果非常差。
任务: 排查全链路原因, 用单一数据集做完整过拟合, 并展示成果。

## 二、全链路排查

### 排查方法

只读分析 `src/` 下五个核心文件, 重点是 teacher forcing、token 设计、推理分段:

- `src/train.py`: 训练循环和 loss
- `src/dataset.py`: 数据加载、mel 切片、token 填充
- `src/maidata_parser.py`: simai ↔ token 转换
- `src/model.py`: Whisper 风格编解码模型
- `src/infer.py`: 推理入口
- `src/mel_cache.py`: mel spectrogram 缓存

### 关键发现 1: teacher forcing 错位

`maidata_parser.to_tensor()` 输出已含 `SOS`:

```python
tokens: list[int] = [SOS]
... append frames ...
tokens.append(EOS)
```

但 `src/train.py` 又额外在 decoder 输入前拼了一个 `SOS`:

```python
dec_input = torch.cat([
    torch.full((tokens.size(0), 1), SOS, ...),
    tokens[:, :-1],
], dim=1)
```

导致目标仍是原始 `SOS`, 模型被训练成 `SOS → SOS`, 推理第一步就吐 `SOS`。用现有 `checkpoints/best.pt` 在真实训练段上验证:

```text
真实前 20 token:  [1, 3, 5, 3046, 3006, ...]
greedy 前 20 token: [1, 3, 5, 3046, 3007, ...]
第一个生成 token: 1 (SOS)
```

### 关键发现 2: 训练/推理分段不一致

训练用 `to_tensor()` 的动态 segment, 段内所有活跃 note 最晚结束时间不超过 30 秒, 在 token 边界切段。

推理用固定滑窗:

```python
WINDOW_FRAMES = 3000
SLIDE_HOP = 2400
```

窗口重叠且和训练分布不匹配。

### 关键发现 3: 缓存路径硬编码

`src/mel_cache.py` 写缓存用全局 `CACHE_DIR`, `relative_to` 用全局 `CHARTS_DIR`, 换 `charts_dir/cache_dir` 参数会失效。

### 关键发现 4: token 长度没问题

抽样 300 谱面:

```text
token_len min/median/p95/max: 8 / 417 / 689 / 1108
over2048: 0
```

`MAX_TOKENS=2048` 不是问题。

## 三、第一批修复

### `src/train.py` 修正 teacher forcing

```python
dec_input = tokens[:, :-1]
target = tokens[:, 1:]
```

loss、mask、accuracy 都改为对齐 `[:, 1:]`, 不再训练模型预测 `SOS`。

### `src/mel_cache.py` 修复硬编码

```python
def process_one(chart_dir, rel_path, cache_dir: Path = CACHE_DIR, ...):
    out = cache_dir / f"{cache_key(rel_path)}.npy"
    ...
    rel = chart_dir.relative_to(charts_dir)  # 传入参数而非全局
```

## 四、单曲过拟合工具 `src/overfit_one.py`

新增独立过拟合脚本, 不改主训练入口:

- 读取一首歌的 `maidata.txt` 和已有 mel cache
- 支持 `--max-segments` 控制段数, `--epochs` 控制训练轮数
- 打印 teacher forcing acc 和 greedy 输出前 80 token
- 用训练 token 经 `parse_from_tensor → generate` 写目标文本, greedy 输出经同样路径写生成文本
- 自动选最佳 checkpoint

### 第一次过拟合结果 (单段)

```text
[overfit] from checkpoint: checkpoints/best.pt
[overfit] device=cuda chart=10. MURASAKi PLUS/夢花火 segments=1 token_len=589
[overfit] epoch=160 loss=0.0068 tf_acc=99.83% greedy_acc=100.00% s0=589/589
```

单段成功 100% greedy match, 但这只是第 1 段。

### 整首 5 段: 起初非常差

```text
segment=0 greedy_match=150/589
segment=1 greedy_match=43/657
segment=2 greedy_match=219/622
segment=3 greedy_match=24/614
segment=4 greedy_match=173/305
parse frames: 262 vs target 346
```

## 五、加入语法约束解码

### `src/constrained_decode.py`

新增 token 语法约束, 让模型只能输出合法结构:

```python
def allowed_tokens(tokens, max_notes_per_frame=33):
    # SOS -> FRAME_START/EOS
    # FRAME_START -> TS
    # TS -> NOTE_TAP/NOTE_TOUCH/NOTE_HOLD/NOTE_SLIDE
    # TAP -> LANE -> 可选 IS_BREAK/IS_EX -> 下一个 note 或 FRAME_END
    # HOLD -> LANE -> TS(duration) -> 可选 IS_BREAK/IS_EX
    # TOUCH -> TOUCH_AREA -> 可选 TS(duration) -> 可选 IS_FIREWORK
    # SLIDE -> 可选 attrs -> 一个/多个 slide segment
    # Slide segment: SEGMENT_START -> SHAPE -> start LANE -> end LANE
    #                GrandV 必须 middle LANE
    #                TS(wait) -> TS(trace) -> 可选段属性 -> SEGMENT_END
    # 段属性 IS_CW/IS_CCW 互斥
    # FRAME_END -> FRAME_START/EOS
```

接入 `src/overfit_one.py` 和 `src/infer.py`, 在 logits 上把非法 token 置为 `-inf`。

### 修过的约束器 bug

- 一开始不允许 GrandV 的 middle lane, 训练集 segment 1 第 8 token 被误判非法
- 训练集存在空 slide (`NOTE_SLIDE` 后直接 `FRAME_END`), 必须放行
- frame 时间戳不能倒退 (训练集没有倒退帧)
- 帧内 note 计数上限先设 8, 后改为训练集实测最大值 33
- 时间戳不能小于上一帧 (允许相等, 因为训练集中存在同一时间点拆多帧)

### 加 `grammar_masked_logits` 训练

`src/overfit_one.py` 训练时也按 `allowed_tokens()` 把非法 token logits 屏蔽, 再算 loss, 让模型训练时就集中在合法 token 上。

### 按 greedy 保存最佳 checkpoint

`src/overfit_one.py` 每 N 轮跑 constrained greedy, 按整首 greedy token 匹配率保存最佳权重, 而不是只看 teacher forcing loss。

### 100% greedy 提前停止

加 `--stop-on-perfect` 和立即保存最佳 checkpoint, 防止中途停止丢失 100% 模型。

## 六、过拟合完成

整首 5 段全部 100%:

```text
[overfit] segment=0 greedy_match=589/589 generated_len=589 target_len=589
[overfit] segment=1 greedy_match=657/657 generated_len=657 target_len=657
[overfit] segment=2 greedy_match=622/622 generated_len=622 target_len=622
[overfit] segment=3 greedy_match=614/614 generated_len=614 target_len=614
[overfit] segment=4 greedy_match=305/305 generated_len=305 target_len=305
[parse_from_tensor] 完成: 346 帧, 5/5 段有效
[overfit] out=tmp/overfit_one_full.txt
[parse_from_tensor] 完成: 346 帧, 5/5 段有效
[overfit] target_out=tmp/overfit_one_target.txt
```

文件级对比:

```text
cmp -s tmp/overfit_one_full.txt tmp/overfit_one_target.txt
cmp_exit=0
```

## 七、推理链路修复

### `src/infer.py` 多处改动

- 接 constrained decode
- 兼容 checkpoint 缺少 `epoch` 字段 (`ckpt.get('epoch', '?')`)
- 加 `decode_segment()` 返回完整 token 列表, 含 `SOS...EOS`, 而非 strip `SOS`
- 加 `fit_window()` 处理窗口 padding/截断
- 加 `--reference-chart` 模式: 用参考谱面的 offset 和长度严格切音频, 做音频到文本过拟合验收

### audio → text 完整闭环 (reference 模式)

```bash
uv run src/infer.py "charts/10. MURASAKi PLUS/夢花火/track.mp3" \
  --checkpoint checkpoints/overfit_one_full.pt \
  --reference-chart "charts/10. MURASAKi PLUS/夢花火/maidata.txt" \
  --out tmp/infer_overfit_one_reference.txt --level 4
```

```text
ref segment 0 offset=1.290s match=589/589 tokens=589
ref segment 1 offset=30.968s match=657/657 tokens=657
ref segment 2 offset=60.484s match=622/622 tokens=622
ref segment 3 offset=90.323s match=614/614 tokens=614
ref segment 4 offset=119.355s match=305/305 tokens=305
[parse_from_tensor] 完成: 346 帧, 5/5 段有效
```

输出文件和 `txt2tensor2txt` target 字节级一致。

## 八、用户继续追问: 滑动推理能用吗?

回答: 普通滑窗在过拟合模型上效果差, 因为训练分段是动态的, 推理用固定 hop 不一致。

## 九、用户: 推理分得更智能一点

讨论了几个方案:

1. 固定窗口训练 → 训推同分布
2. 预测/复用动态 segment → 更复杂
3. “解码多少就切” → 跟进模型输出决定切段
4. 重叠上下文 + 中心提交 → ASR 里常用

最终用户选了第四种: 上下文窗口重叠, 只提交中心区域, 不和训练一样切干净。

## 十、overlap 推理实现

### `infer.py` overlap 模式

每个窗口:

```text
window = 30s  (3000 mel frames)
context = 5s (左/右各 5s)
stride = window - 2 * context = 20s
```

每个窗口:

```text
[left 5s | commit 20s | right 5s]
```

只保留 `commit_start ≤ abs_time < commit_end` 的 frame。

### 去重

```python
key = (round(time_sec / 0.02), notes_repr)
score = -abs(abs_time - center)
同 key 保留 score 高的 (离窗口中心近)
```

### 新增函数

- `overlap_infer()`: 滑动 + 中心提交 + 去重, 返回全局 frame 列表
- `frames_to_maidata()`: 从 frame 列表生成 maidata 文本

### CLI

```text
--mode overlap|fixed
--context-sec 5.0
--start-sec 0.0
```

### 结果

```text
overlap total frames: 189
target frames: 346
precision 0.0106
recall 0.0058
```

不好。原因: 训练分段是 `[0.0, 28.7, 57.4, 87.7, 117.3]`, overlap 窗口是 `[0.0, 24.8, 49.6, 74.5, 99.3, 124.1]`, 后续窗口不是训练分段位置, 模型没有见过这种任意 offset。

## 十一、用户: 训练从 0 秒开始, 不应该跳过静音吗?

指出训练切段设计问题: 第一段从第一帧 note 的 1.29s 开始, 真实音频自然从 0s 开始, 分布不一致。

### 修复: `maidata_parser.to_tensor()`

```python
seg_offset = 0.0  # 第一段从音频 0 秒开始
# 后续段仍按谱面 frame 起点动态切分, 避免训练/原始音频推理起点不一致
```

### 重新训练

新 offsets:

```text
[0.0, 28.710, 57.419, 87.742, 117.258]
```

零起点重训:

```text
epoch=140 loss=0.0265 tf_acc=99.21% greedy_acc=85.61% s0=509/543 s1=389/614 ...
```

继续微调:

```text
epoch=040 loss=0.0015 tf_acc=100.00% greedy_acc=100.00%
s0=543/543 s1=614/614 s2=665/665 s3=629/629 s4=336/336
[overfit] greedy 已 100%, 提前停止
```

### 零起点 reference 推理

```text
ref segment 0 offset=0.000s match=543/543
ref segment 1 offset=28.710s match=614/614
ref segment 2 offset=57.419s match=665/665
ref segment 3 offset=87.742s match=629/629
ref segment 4 offset=117.258s match=336/336
```

零起点 overlap 推理仍不好, 因为模型仍然是强过拟合, 没学过任意 offset 窗口。

## 十二、用户: 加上更严格的约束

### 1. 相邻帧时间差不能太近

全量统计 10ms token 单位:

```text
gap 0: 438
gap 1: 466
gap 2: 772
gap 3: 3381
gap 4: 11992
gap 5: 12179
```

相邻帧 0.01s/0.02s 不算异常, 不能直接禁掉, 否则会拦训练集。安全规则仍是: 不允许倒退, 允许相等。

### 2. TAP/HOLD/SLIDE 合计上限

全量统计:

```text
TAP+HOLD+SLIDE per frame:
0: 72865
1: 1302327
2: 521734
3: 5
4: 5
```

最大值 4, 几乎都是 ≤2。

### 收紧约束器

```python
def allowed_tokens(tokens, max_notes_per_frame=33, max_action_notes_per_frame=4):
```

规则:

- 每帧 TAP/HOLD/SLIDE 合计最多 4 个 (TOUCH 不计入此限制)
- 每帧总 note 数仍最多 33 个

加入 `_allowed_in_frame` 计数:

```python
action_notes = 0
# 消费 note 时计数
if tok in (NOTE_TAP, NOTE_HOLD, NOTE_SLIDE):
    action_notes += 1
# 达到上限时只允许 TOUCH 或 FRAME_END
if action_notes >= max_action_notes_per_frame:
    return (NOTE_TOUCH, FRAME_END)
```

### 全量验证

```text
files_checked 1687
segments_checked 30652
constraint_violations 0
timestamp_decreases 0
max_notes_per_frame 33
max_tap_hold_slide_per_frame 4
```

约束器不拦训练集, 时间戳不倒退。

## 十三、最终交付

### 修改/新增文件

```text
src/train.py                 # teacher forcing 修正
src/overfit_one.py           # 单曲过拟合工具
src/constrained_decode.py    # 语法约束解码器
src/infer.py                 # 接约束解码 + reference 模式 + overlap 模式
src/mel_cache.py             # 缓存路径硬编码修复
src/maidata_parser.py        # 第一段 offset 从 0 秒开始
```

### 关键产物

```text
checkpoints/overfit_one_full.pt   # 零起点整首过拟合 100% greedy match
tmp/overfit_one_target.txt        # 训练 token 还原参考文本
tmp/overfit_one_full.txt          # 过拟合 greedy 输出文本 (与 target 字节级一致)
tmp/infer_overfit_zero_reference.txt  # 音频→文本 reference 推理输出
tmp/infer_overfit_overlap_zero.txt    # 音频→文本 overlap 推理输出
```

### 三种推理模式对比

```text
reference:  346/346 帧 匹配, 字节级一致
overlap:    189 帧, precision 0.0106, recall 0.0058
fixed:      不推荐
```

### 已知限制

- 普通 overlap 推理仍不好, 因为模型只过拟合了训练分段位置
- 若要 overlap 泛化可用, 需要把训练也改成 overlap-window 数据集
- 当前推理时间策略: 训练/参考分段从 0 秒开始, 时间戳单调非递减
- 语法约束能保证结构合法, 不能保证音乐内容合理

## 十四、后续可做

1. `ChartDataset` 增加 overlap-window 数据集, 训推同分布
2. 增加边界预测 token (例如 `NEXT_TS_x`), 让模型自己决定 next offset
3. 自适应 beam search / length prediction, 让推理不依赖 reference
4. 多曲过拟合 + 多曲泛化评估

## 十五、改成滑窗训练后的整曲过拟合验证

用户要求: “改成滑动着训。”

目标不是先追求泛化, 而是验证新的训练/推理定义是否自洽:

```text
同一首歌的所有滑窗样本可以被模型记住
然后必须只用 src/infer.py 推理这首歌
推理出的谱面内容要和训练目标一致
```

### 1. `src/dataset.py`: 从动态分段改成固定滑窗样本

之前的训练样本来自 `compiler.to_tensor()` 的动态 segment:

```text
输入: 某个谱面动态 segment 对应的 mel 片段, 再 pad/trim 到模型长度
目标: 这个动态 segment 的 token
```

这和 `infer.py --mode overlap` 的滑窗推理不一致。推理时窗口是按固定步长切的, 不会刚好落在训练的动态 segment 起点。

因此把 `compile_index()` 改成直接构造滑窗样本:

```text
输入窗口: 固定 3000 个 mel 帧
提交区: 每 20 秒一个窗口
标签: 只包含提交区内“开始”的音符
跨边界 hold/slide: 不拆开, 归属于开始时间所在的提交区
```

实际窗口时长不是严格 30 秒。当前音频参数是:

```text
sample_rate = 22050
hop_length = 256
3000 mel frames = 3000 / (22050 / 256) = 34.8299s
```

所以默认配置实际是:

```text
左上下文: 5.0s
提交区: 20.0s
右上下文: 约 9.83s
输入长度: 3000 mel 帧
```

为什么这样改:

```text
模型结构仍要求 encoder 输入 3000 个 mel 帧
如果强行改成真实 30 秒, 还要一起改 hop/window/model ctx
当前最小正确改法是保持 3000 帧, 让训练和推理都用同一套滑窗定义
```

关键实现点:

```python
commit_start = 0.0
while commit_start < song_end_sec:
    commit_end = min(commit_start + commit_sec, song_end_sec)
    input_start = min(
        max(0.0, commit_start - left_context_sec),
        max(0.0, song_end_sec - window_sec),
    )
    start_frame = int(input_start * frames_per_sec)
    end_frame = min(mel_total_frames, start_frame + mel_frames)
```

目标 token 只收提交区内开始的 frame:

```python
if not (commit_start <= frame.time_sec < commit_end):
    continue
```

### 2. `src/dataset.py`: token 时间戳使用实际切片帧对应时间

第一次整曲验证时, 推理输出大量逗号位置微小偏移, 最后还少边界帧。

原因是训练样本先用浮点 `input_start` 计算 token 时间戳, 再用整数 `start_frame` 切 mel:

```text
token offset 基于 input_start
mel 实际切片基于 int(input_start * frames_per_sec)
```

这两者最多差不到 1 个 mel 帧, 但 token 时间戳是 10ms 量化, 会影响生成文本里的逗号数量和边界帧归属。

修复: token 时间戳改成基于实际切片帧:

```python
actual_input_start = start_frame / frames_per_sec
tl.append(c._ts_token(frame.time_sec - actual_input_start))
```

为什么这样改:

```text
模型看到的音频从 start_frame 开始
token 时间也必须相对 start_frame 对应的真实时间
训练和推理都必须使用同一个 offset 定义
```

### 3. `src/infer.py`: overlap 推理改成和训练同分布

推理原来的 overlap 逻辑是根据窗口自身计算提交区, 和新训练定义不完全一致。

改成显式参数:

```text
--context-sec 5
--commit-sec 20
```

推理窗口逻辑:

```text
commit_start = 0, 20, 40, ...
input_start = commit_start - context_sec, 但靠近开头/结尾时夹到合法范围
输入 mel = input_start 对应帧开始, 连续取 3000 帧（约 34.83 秒, 不是严格 30 秒）
只提交 [commit_start, commit_start + commit_sec) 内的 frame
```

同时修复 offset, 使用实际 mel 切片帧对应时间:

```python
pos = int(input_start * frames_per_sec)
offset_sec = pos / frames_per_sec
```

为什么这样改:

```text
训练 token 的时间戳相对实际 start_frame
infer 解析生成 token 后加回的 offset 也必须是实际 pos 对应时间
否则会出现 10ms 量化级别的偏移
```

### 4. `src/infer.py`: 提交区左边界放宽 10ms

整曲过拟合达到 100% 后, 用 `infer.py` 推理仍少 3 帧:

```text
60.0s
80.0s
120.0s
```

这些帧正好在提交区起点边界。

原因:

```text
训练按 [commit_start, commit_end) 归属 frame
token 时间戳是 10ms 量化
推理时 abs_time 浮点还原后可能落在 commit_start 左侧极小误差处
硬过滤 commit_start <= abs_time 会丢边界帧
```

修复: 左边界放宽 10ms, 右边界仍保持排他:

```python
boundary_tol = 0.01
if not (commit_start - boundary_tol <= abs_time < commit_end):
    continue
```

为什么只放宽左边界:

```text
边界帧归属新窗口
左边界需要容忍 10ms 量化误差
右边界保持排他, 避免把下一提交区的 frame 提前收进来
后续还有 key 去重, 不会因为边界容差产生重复 note
```

修复后推理帧数从 381 回到 384。

### 5. `src/overfit_window.py`: 单滑窗样本过拟合验证

新增脚本:

```text
src/overfit_window.py
```

用途:

```text
取 ChartDataset 的一个滑窗样本
只训练这个样本
用 constrained greedy 自回归生成
把生成 token 和目标 token 都转成 maidata 文本比较
```

验证命令:

```bash
PYTHONPATH=src uv run src/overfit_window.py --idx 0 --epochs 600 --eval-every 50 --lr 1e-3 --stop-on-perfect
```

结果:

```text
看答案续写=100.00%
自己生成=100.00%
match=289/289
```

文本比较:

```text
tmp/overfit_window.txt
tmp/overfit_window_target.txt
cmp 结果: MATCH
```

结论:

```text
单个滑窗样本的训练、loss、解码、token 解析、文本生成链路是通的
```

### 6. `src/overfit_song.py`: 整首歌滑窗过拟合验证

新增脚本:

```text
src/overfit_song.py
```

用途:

```text
用某个样本所在歌曲的所有滑窗窗口训练
保存 checkpoints/overfit_song.pt
输出目标文本 tmp/overfit_song_target.txt
然后必须用 src/infer.py 对原 track.mp3 推理
```

本次使用的歌曲:

```text
audio: charts/00. UTAGE/[奏] マイオドレ！舞舞タイム [2P]/track.mp3
chart: charts/00. UTAGE/[奏] マイオドレ！舞舞タイム [2P]/maidata.txt
windows: indices=[0, 1, 2, 3, 4, 5, 6]
```

训练命令:

```bash
PYTHONPATH=src uv run src/overfit_song.py --idx 0 --epochs 500 --eval-every 25 --lr 1e-3 --batch-size 2 --stop-on-perfect
```

结果:

```text
看答案续写=100.00%
自己生成=100.00%
保存最好模型到 checkpoints/overfit_song.pt
```

### 7. 必须用 `src/infer.py` 推理整首歌

推理命令:

```bash
PYTHONPATH=src uv run src/infer.py "charts/00. UTAGE/[奏] マイオドレ！舞舞タイム [2P]/track.mp3" \
  --checkpoint checkpoints/overfit_song.pt \
  --out tmp/overfit_song_infer.txt \
  --level 4 \
  --mode overlap \
  --context-sec 5 \
  --commit-sec 20
```

最终推理日志:

```text
overlap window 0 ... commit 0.0..20.0s frames=44 kept=44
overlap window 1 ... commit 20.0..40.0s frames=47 kept=47
overlap window 2 ... commit 40.0..60.0s frames=62 kept=62
overlap window 3 ... commit 60.0..80.0s frames=61 kept=61
overlap window 4 ... commit 80.0..100.0s frames=66 kept=66
overlap window 5 ... commit 100.0..120.0s frames=59 kept=59
overlap window 6 ... commit 120.0..139.0s frames=45 kept=45
total frames: 384
```

### 8. 最终比较结果

目标文本:

```text
tmp/overfit_song_target.txt
```

推理文本:

```text
tmp/overfit_song_infer.txt
```

字节级比较仍不同, 但唯一差异是 metadata:

```diff
-&title=default
+&title=generated
```

按解析后的谱面 frame/note 集合比较:

```text
target frames: 384
infer frames: 384
missing: 0
extra: 0
```

结论:

```text
整首歌滑窗过拟合成功
只用 src/infer.py 推理同一首歌, 谱面内容完全对齐
当前训练/推理滑窗定义已经自洽
```

### 9. 这次留下的代码改动汇总

```text
src/dataset.py
  - compile_index 改成固定滑窗样本
  - 目标只包含提交区内开始的音符
  - token 时间戳基于实际 start_frame 对应时间
  - 保持输入 3000 mel 帧

src/infer.py
  - overlap 推理增加 --commit-sec
  - commit_start 按固定步长推进
  - offset_sec 基于实际 pos 对应时间
  - 提交区左边界增加 10ms 容差

src/overfit_window.py
  - 新增单滑窗样本过拟合验证脚本

src/overfit_song.py
  - 新增整首歌滑窗过拟合验证脚本
  - 支持从 checkpoint 续训
  - 输出训练目标文本和 infer.py 推理命令
```

### 10. 对后续训练的意义

这次验证只能证明一件事:

```text
滑窗训练样本定义、模型训练、自回归解码、infer.py overlap 推理链路是自洽的
```

它不证明全量训练一定能泛化。全量训练如果仍生成垃圾, 下一步应该看:

```text
模型容量是否够
训练轮数是否够
学习率/正则是否合适
数据规模和难度是否支持从音频泛化到谱面
内容指标是否真的上升
```

但至少已经排除了:

```text
滑窗标签和 infer.py 推理窗口定义不一致
边界 frame 被系统性丢掉
单样本/单曲无法通过自回归生成还原目标
```

## 十六、8 方向旋转数据增强

用户要求: Tap、Hold、Touch、Slide 都在 8 轨圆盘上整体右转, 最多转 7 次, 第 8 次回到原谱面。训练数据集扩大 8 倍。

### 1. 增强位置

增强放在 `src/dataset.py` 的 token 层:

```text
音频 mel 不变
token 里的轨道位置右转
```

这样不需要重新 parse maidata, 不需要改模型, 也不影响推理脚本。

### 2. 旋转规则

`LANE_1 ~ LANE_8` 循环右转:

```text
LANE_i -> LANE_((i - 1 + steps) % 8 + 1)
```

覆盖范围:

```text
Tap lane
Hold lane
Slide start lane
Slide end lane
GrandV middle lane
```

Touch 分组循环:

```text
A1~A8: 循环右转
B1~B8: 循环右转
C: 不变
D1~D8: 循环右转
E1~E8: 循环右转
```

Slide shape 和方向属性不变:

```text
SLIDE_SHAPE_* 不变
IS_CW / IS_CCW 不变
```

原因:

```text
整体旋转不会改变相对形状
整体旋转不是镜像, 不交换 p/q、pp/qq、s/z
顺时针/逆时针在整体旋转后仍然是顺时针/逆时针
```

### 3. 新增代码

`src/dataset.py` 新增:

```text
rotate_token_id(token, steps)
rotate_token_list(tokens, steps)
RotatedDataset(base, rotations=8)
```

`RotatedDataset` 的索引规则:

```text
base_idx = idx // 8
rotation = idx % 8
```

所以训练集大小变成:

```text
len(train_base) * 8
```

### 4. 训练集增强, 验证集不增强

`src/train.py` 中只包装训练集:

```text
base_train_ds = Subset(full_ds, train_indices)
train_ds = RotatedDataset(base_train_ds)
val_ds = Subset(full_ds, val_indices)
```

验证集保持原始方向, 避免验证指标被增强分布污染。

训练日志会打印:

```text
Train: 原始 windows / charts, 旋转增强后 samples, Val: windows / charts
```

### 5. 自检

新增脚本:

```text
src/check_rotation.py
```

检查内容:

```text
rotate(tokens, 0) == tokens
rotate(tokens, 8) == tokens
连续右转 1 次执行 8 遍后 == tokens
Touch C 旋转 0~7 次都不变
每个 rotation=0..7 的 token 都能被 parser 解析
```

运行命令:

```bash
PYTHONPATH=src uv run src/check_rotation.py
```

结果:

```text
[rotation] 旋转增强自检通过
```

### 6. 注意事项

这是真 8 倍数据集, 不是随机增强:

```text
每个 epoch 训练样本数约 8 倍
每个 epoch 训练时间也约 8 倍
```

后续如果训练太慢, 可以再改成“每次随机旋转”的在线增强, 但当前实现严格符合“数据集直接翻 8 倍”的要求。

### 11. 关于“30 秒”和“3000 mel 帧”的澄清

`doc/token化设计.md` 里的 30 秒约束指的是 token 时间戳范围:

```text
TS_0 ~ TS_2999
100fps
最多表达 30 秒相对时间
```

当前工程里的模型输入窗口不是严格 30 秒，而是:

```text
n_audio_ctx = 1500
mel_frames = 2 * n_audio_ctx = 3000
sample_rate = 22050
hop_length = 256
3000 / (22050 / 256) ≈ 34.83 秒
```

所以推理日志出现下面这种输出是符合当前实现的:

```text
overlap window 0 0.0..34.8s commit 0.0..20.0s
```

这不等于 token 片段超过 30 秒。当前训练目标只覆盖 20 秒提交区, token 相对时间仍在 `TS_0~TS_2999` 可表达范围内。

如果以后要让音频输入窗口也严格等于 30 秒, 需要同步改模型输入长度和所有训练/推理脚本, 旧 checkpoint 不能复用。
## 当前设计问答和结论

### 澄清当前的设计模式

问：当前训练集是不是固定扩大了 8 倍？

答：是。当前训练集用 `RotatedDataset` 固定扩大 8 倍，只包装训练集，验证集不增强。只做 8 轨整体旋转：顺时针 `0..7` 次已经覆盖所有圆周旋转。逆时针 1 次等价于顺时针 7 次，不会增加新样本。如果只做整体旋转，最多就是 8 倍。

问：当前滑窗训练是怎么组装 token 张量的？

答：训练窗口按目标绝对起点 `0, 1, 2...` 每秒平移，逻辑音频窗口起点是 `target_start - 12s`。输入固定为 `2612` 个 MERT 帧；token 包含相对 `6..12s` 的真实谱面前缀和相对 `12..22s` 的目标。`loss_mask` 只监督目标区完整 frame token 与 EOS。

问：缓存索引会不会在 token、mask 或坐标传递时弄坏训练样本？

答：当前缓存路径已和旧的内存编译路径逐窗口差分验证：真实 Master 谱面的 token、`loss_mask`、音频切片坐标、padding、8 方向旋转、`collate`、双 worker `DataLoader` 以及 `tokens[:, :-1] -> tokens[:, 1:]` teacher forcing 对齐均一致。缓存仅把 token 序列压为 `uint16`、把连续 loss mask 压为 `loss_start`；读取时恢复为独立 `int64` tensor 和布尔 mask。改动 `chart_cache.py`、滑窗边界、token 词表、mask 语义或数据加载转换时，必须重新做旧/新路径的逐元素差分，至少覆盖有 Slide、空目标窗口和最大 token 长度边界的真实谱面。

问：索引缓存如何自动重建且不影响并发训练？

答：`ChartDataset` 初始化会自动检查 Mel 和谱面索引，无需手动运行脚本。正常启动只对源文件做 `stat` 校验；任一 `maidata.txt`、Mel 文件或缓存配置变化时才重新解析并构建。新索引写到新的 generation 目录，完成后原子替换 `current.json`，旧 generation 不会在运行中删除。锁文件记录 PID 和创建时间，异常退出留下的死锁会被回收。`size + mtime_ns` 不防御刻意伪造相同元数据的内容篡改；需要时应手动清理对应缓存目录强制重建。

问：当前滑窗推理是模型生成到 EOS 就切下一个窗口吗？

答：不是。正式推理固定按 `commit_start += 10s` 推进，EOS 只结束当前窗口。第一窗从 `SOS` 生成；后续窗口把已经生成并提交的最后 6 秒 frame 重新编码为当前相对 `6..12s` 的 token 前缀，再生成相对 `12..22s` 的新目标并只提交这 10 秒。

问：当前滑窗推理有没有正确处理时间戳？

答：有。token 里的 TS 是相对逻辑窗口起点 `window_start = target_start - 12s`，不是绝对时间；头尾补零不会改变这个起点。推理时 `_parse_token_segment()` 先得到相对时间，再用 `abs_time = window_start + frame.time_sec` 加回绝对时间。同一个真实 23 秒的 note，在 `window_start=0` 的窗口里是 `TS_23s`，在 `window_start=15s` 的窗口里是 `TS_8s`。

问：当前有没有真正用上 KV cache？

答：没有真正的 decoder KV cache。`model.py` 有 `kv_cache` 参数，但项目里没有调用路径传入 cache，也没有完整的 cache hook。当前 `infer.py` 已经在每个窗口只调用一次 `model.embed_audio(mel)`，后续每步复用 audio features 并重新运行完整 decoder prefix；因此 encoder 没有重复跑，decoder self-attention 仍然会重复计算。

问：训练的时候能不能用 KV cache？

答：训练时不应该用 KV cache。teacher forcing 一次 forward 并行预测所有位置，比逐 token cache 更适合训练。KV cache 主要用于自回归推理。当前已完成每个窗口只跑一次 `model.embed_audio(mel)` 并复用 audio features；后续如实现 decoder KV cache，需要正确区分 self-attention cache 和 cross-attention/audio cache，不能随便传一个空 dict。
### 对于进一步改进的疑问

问：`2612 MERT frames ≈ 34.83s` 是不是当前滑窗方案的严重问题？

答：对当前实现不是。超过提交区的右侧部分只是上下文，不会被提交，也不会进入训练目标。窗口起点固定为 `target_start - 12s`，歌曲尾部通过右补零处理，不会再夹动窗口；目标时间戳始终位于相对 `12..22s`，不会因尾窗超过 `29.99s`。不要武断优先大改窗口长度；应先依据整曲评估确认边界质量是否是瓶颈。

问：要不要改成 `10s 左上下文 + 10s 提交区 + 10s 右上下文`？

答：这是合理方案，能让提交区更居中，边界更稳。如果保留当前 `2612` 个 MERT 帧，实际会变成 `10s 左上下文 + 10s 提交区 + 约 14.83s 右上下文`。代价是窗口数和推理次数大约翻倍；再叠加 8 倍旋转，训练成本会明显增加。如果边界质量差，可以考虑改；但应该和整曲评估一起看，不要盲目增加计算量。

问：左边音频上下文里，模型能不能看到已经生成过的 token？

答：能看到其中最后 6 秒。训练 token 包含相对 `6..12s` 的真实谱面前缀，但只对相对 `12..22s` 的目标区计算 loss；推理时会把前一窗口已经提交的最后 6 秒 frame 重新编码为当前窗口前缀。首窗才从 `SOS` 开始。前缀会带来跨窗口连续性，也意味着推理中的前缀错误可能传递到后续窗口；这与训练使用真实前缀的分布存在差异。

问：滑窗要不要改成按 1 秒步进，让同一段 30 秒音频用不同 offset 训练很多遍？

答：当前训练已经按 1 秒步进，让同一段音频以不同相对时间出现在多个窗口中，因此已经覆盖时间平移等变性的主要收益；再额外加密步长只会显著增加高度重复的样本。训练窗口从 `target_start=0,1,2...` 构造，正式推理固定每次提交 10 秒。只有整曲评估证明固定相位或边界仍是瓶颈时，才考虑改变训练采样策略。

问：当前约束解码和要求大语言模型格式化输出 JSON 是同类方法吗？

答：是同一类思想，都是 grammar-constrained decoding。`constrained_decode.py` 的 `allowed_tokens()` 在推理时每一步先拿模型 logits，再把非法 token 置为 `-inf`，只在合法集合里选下一个 token。当前实现不仅约束字段顺序，还会重放同 TS 资源、持续 HOLD/TOUCH_HOLD、Slide 几何和 no-head 等状态；返回的候选应避免已知必然走入空 allowed set 的路径。它仍不能保证谱面跟音乐对齐、难度合理或有音乐性。如果 parser 一直报警告或推理出现 `grammar_dead_end`，优先检查 `allowed_tokens()` 的候选可完成性以及它与 `_parse_token_segment()` 的语义是否一致。

问：训练时能不能也用 `allowed_tokens()` 约束着训？

答：可以，但当前主训练 loss 没有这么做。主训练是普通 teacher forcing：完整词表上预测下一个 token。`allowed_tokens()` 目前主要用于推理、验证里的 greedy decode、过拟合脚本的生成路径。如果要做约束训练，先验证所有训练 target 在对应 prefix 下都属于 allowed set，否则约束器会把正确答案屏蔽掉。建议先加可选开关，例如 `MAIMAI_GRAMMAR_MASK=1`，不要默认启用。朴素实现需要对 batch 里每个位置跑 Python 状态机，可能很慢。


## 当前设计问题的改进

问：当前损失函数能不能正确评估训练集和验证集的效果？

原回答：当前 `CrossEntropyLoss(ignore_index=PAD, label_smoothing=...)` 能正确评估“精确 token 序列预测”的 train/val loss，但不能正确反映“时间对了、音符类型对了、路数无所谓”的谱面效果。`val_acc`、`greedy_accuracy` 都要求 token 完全一致，原来的 `content_match_counts()` 也使用 `repr(note)`，会把 lane、Touch 区域和 Slide 起终点不同判成错误。现有 loss 可以保留为精确 token 复现的辅助指标，但不能单独用来判断生成内容是否正确。

解决方案：已新增 `src/content_metrics.py`，将生成 token 和目标 token 解析为音符事件，进行一对一匹配并计算 precision、recall、F1，`src/train.py` 已改为使用该指标。匹配忽略 Tap/Hold 的 lane、Touch 的具体区域、Slide 的起点/终点/GrandV 中间 lane，以及普通音符的 Break、EX 和 Slide 的 `isSlideBreak`。所有音符必须在 `10ms` 时间容差内且 `NoteType` 一致；Hold 和 Touch Hold 还要匹配持续时间；Slide 还要逐段匹配 `SlideShape`、等待时长、滑动时长、`isClockwise`、`isForceStar`、`isFakeRotate` 和 `isSlideNoHead`。持续时间按 token 的 `10ms` 精度比较，重复预测不能重复命中同一个目标事件。

问：为什么整曲生成几乎每个 10 秒窗口都达到 2048 token，却不生成 EOS？

原回答：不能直接归因于模型生成 slop。需要先检查 EOS 是否进入 loss、是否被 grammar 屏蔽、是否在 teacher forcing 下能预测，以及真值 token 前缀和模型递推前缀下的终止行为是否不同。

解决方案：已对 3 首过拟合歌曲的 318 个训练窗口逐 token 重放 `allowed_tokens()`。64650 个受监督 token 中没有 grammar 违规，318 个 EOS 全部合法且都紧跟 `FRAME_END`，排除了标签错位和约束器屏蔽。统计发现 EOS 只占受监督 token 的 `0.492%`；在 `FRAME_END` 后的二选一决策中，继续 `FRAME_START` 有 8173 次，EOS 只有 318 次，结束率仅 `3.75%`。普通交叉熵下，训练早期始终选择继续下一帧是明显的类别失衡局部最优。当前默认给 EOS 使用 `5x` loss 权重，可通过 `MAIMAI_EOS_LOSS_WEIGHT` 调整。验证日志已新增 EOS 准确率、平均概率、平均排名、误判为 `FRAME_START` 的比例；另用少量真值前缀窗口与正式递推整曲推理对照，并汇总两种模式的撞上限比例和平均新增 token 数。若真值前缀也撞上限，问题在 EOS 学习；若只有正式递推撞上限，问题在模型生成前缀造成的误差传播。
