# 模型架构潜在问题评估

从 `model.py` 的张量流和注意力实现入手，再交叉核对训练、数据窗口与推理调用，重点找训练/推理不一致、容量瓶颈、掩码错误、数值稳定性和性能问题。全程只读，不使用子代理，也不修改文件。

目前架构本质上是“冻结 MERT 特征 + 仅 LayerNorm/绝对位置编码 + 标准自回归 Transformer decoder”，音频侧没有可学习的时序编码器。下一步我会核对实际超参数、MERT 特征来源和 token 长度/密度，判断主要风险究竟是结构表达力、注意力计算规模，还是训练目标与生成过程的错配。

**总体判断**
当前模型能作为谱面生成基线工作，但架构上存在一个明显的不平衡：

```text
冻结 MERT-v1-95M
    ↓ 2612 × 768
逐帧 LayerNorm + 绝对位置编码
    ↓
4 层、768 维自回归 Transformer decoder
    ↓
最长 2048 token
```

音频侧几乎没有可训练的建模，文本侧却承担了音频重整、时序对齐、节奏抽象、谱面规划、语法生成和窗口终止的全部任务。结合当前 1 秒高重叠训练窗口、真实谱面前缀 teacher forcing 和递归推理方式，我认为最值得优先关注的不是单纯“层数太少”，而是以下问题。

## 高优先级问题

### 1. 训练与正式推理存在严重的前缀分布偏移

训练时，decoder 看到的 6 秒谱面前缀始终来自真值：

- 数据构造：`src/dataset.py:247-262`
- 前缀不计算 loss：`src/dataset.py:252-260`
- teacher forcing：`src/train.py:260-272`

正式推理时，前缀来自模型前一个窗口已经生成的结果：

- `src/infer.py:240-249`

一旦前一窗口出现漏帧、时间偏移、错误类型或密度异常，这些错误就会作为后续窗口的条件继续传播。模型训练中从未学习“如何从有缺陷的历史谱面继续生成”。

这不仅是普通 token 级 exposure bias。它同时发生在两个层面：

1. 同一窗口内，训练看真值 token，推理看自身 token。
2. 跨窗口时，训练看真值谱面前缀，推理看上一窗口生成谱面。

现有 `validate_oracle_prefix()` 与正式整曲推理对照能够观察这个问题，但不能消除它，见 `src/train.py:402-449`。

**可能表现**

- 真值前缀生成明显好于正式整曲生成。
- 前几个窗口尚可，歌曲后半段逐渐退化。
- 某个窗口密度或节奏出错后，后续窗口持续偏离。
- teacher-forcing loss 很低，但整曲 F1 提升有限。

**优先验证**

记录同一批窗口的三组结果：

1. 真值 token 全 teacher forcing。
2. 真值 6 秒前缀后自回归生成。
3. 模型上一窗口前缀后自回归生成。

如果 2 明显差于 1，是窗口内 exposure bias；如果 3 明显差于 2，是跨窗口前缀分布偏移。

---

### 2. 音频侧缺少可训练的时序适配器

`embed_audio()` 只做逐帧 LayerNorm 和绝对位置编码：

```python
return self.audio_ln(features) + self.audio_position.to(features.dtype)
```

见 `src/model.py:229-251`。

它没有：

- 音频侧 self-attention。
- 局部卷积或时间聚合。
- 降采样。
- MERT 到谱面任务空间的非线性投影。
- 显式节拍或 onset 表征。
- 多尺度时序建模。

因此，4 个 decoder block 中的 cross-attention 必须直接从 2612 个冻结 MERT 帧中完成全部任务相关抽象，见 `src/model.py:147-164`。

MERT 的最终隐藏层不一定是最适合谱面生成的层。它可能更擅长音乐语义、音色和高层内容，而谱面生成高度依赖：

- 瞬态和 onset。
- 局部节奏周期。
- 拍点与小节结构。
- 强弱变化。
- 乐句边界。
- 多时间尺度重复。

冻结 MERT 并非问题本身，问题是冻结特征和 decoder 之间没有廉价的可训练转换层。

**潜在后果**

- decoder 花费大量容量学习如何解释 MERT，而不是学习谱面结构。
- cross-attention 难以形成稳定的单调时间对齐。
- 对训练歌曲过拟合容易，对新歌曲泛化差。
- 模型可能更多依赖谱面 token 前缀，而不是音频。
- 时间遮挡增强会比预期更伤，因为相邻特征没有可训练的音频侧重建能力。

**建议的最小架构实验**

不要先恢复大型音频 Transformer。先测试一个很小的残差适配器：

```text
LayerNorm
→ Linear(768, 768)
→ GELU
→ Linear(768, 768)
→ residual
→ audio position
```

若这个改动就显著提高验证集整曲 F1，说明冻结特征与任务空间错配确实是主要瓶颈。只有适配器不够时，再测试 1 到 2 层局部卷积或音频 self-attention。

---

### 3. 模型缺乏明确的“目标区结束”状态

每个窗口负责生成相对 `12..22s`，但 token 序列中没有显式的目标边界或剩余时间信号。模型只能从以下信息间接判断何时输出 EOS：

- 已生成 frame 的时间戳。
- 音频内容。
- 文本绝对位置。
- 训练数据中 EOS 的统计规律。

EOS 只出现在最后一个目标 frame 后；如果目标区后半段没有音符，最后一个时间戳可能离 22 秒很远。模型必须学会：

> 当前最后一个 frame 虽然在 17 秒，但 17 到 22 秒已经没有需要生成的事件，因此现在应结束。

这比常规语言模型的句末判断困难，因为结束条件主要来自未来音频中“事件不存在”。当前约束器只保证 EOS 在语法上可用，不能告诉模型何时应该结束。

已有统计显示 EOS 极不平衡，而且模型倾向在 `FRAME_END` 后继续生成 `FRAME_START`。5 倍 EOS 权重只是调整分类代价，未解决结束状态表示不足。

**潜在表现**

- 撞 2048 token 上限。
- 在最后一个真实事件后继续编造 frame。
- 空目标窗口不能立即 EOS。
- EOS 准确率对 loss 权重非常敏感。
- 提高 EOS 权重后又出现过早结束。

**更根本的方向**

优先考虑目标结束机制，而不是继续盲调 EOS 权重：

- 增加固定的目标结束时间条件。
- 在每个 frame 后让模型明确感知距离 22 秒还有多远。
- 将 EOS 约束为“没有更晚事件”决策，而不是与所有普通 token 混在一起学习。
- 或把生成改为按离散时间槽推进，使终止由时间轴自然决定。

这是 token/目标设计与模型架构的交界问题，预计比单纯增加 decoder 层数更重要。

---

### 4. 没有任何 dropout，泛化风险较高

当前注意力、MLP、embedding 和 residual 分支均没有 dropout：

- 注意力：`src/model.py:53-126`
- MLP：`src/model.py:141-144`
- decoder：`src/model.py:167-226`

当前数据又高度相关：

- 训练窗口每 1 秒移动一次。
- 相邻窗口约重叠 33.83 秒音频。
- 每个基础窗口复制成 8 个旋转版本。
- 8 个版本共享完全相同的音频。
- 当前配置甚至是 5 首歌曲过拟合模式，见 `config.yaml:85-90`。

因此样本数量看起来很大，但独立音乐内容远少于样本数。43.7M 可训练参数在这种数据结构下很容易记忆歌曲和局部窗口。

已有特征时间遮挡可以提供一定正则，但它不能替代网络内部正则，而且当前只遮挡最多约 0.5 秒。

**需要区分**

- 5 首歌过拟合实验中不应加 dropout，否则会干扰过拟合诊断。
- 切回正常全量训练后，缺少 dropout 才是需要重点验证的问题。

建议正常训练时先测试较小值，例如 residual/attention dropout `0.05`，而不是直接使用 `0.1` 或更高。

## 中高优先级问题

### 5. 时间戳和持续时间使用同一组 3000 个独立类别

时间 token 为：

```text
TS_BASE ... TS_BASE + 2999
```

既用于：

- frame 相对时间。
- Hold 持续时间。
- Slide 等待时间。
- Slide 滑动时间。

见 `src/tokenizer.py:14-16`、`src/tokenizer.py:70-114`。

从模型角度看，`12.34s` 和 `12.35s` 是两个独立 embedding，没有天然的相邻、大小或距离关系。更重要的是，同一个 token 在不同上下文中可能表示：

- frame 发生在窗口第 12.34 秒。
- Hold 持续 12.34 秒。
- Slide 等待 12.34 秒。

语法上下文能帮助区分语义，但共享 embedding 仍要求模型自己学习“绝对位置”和“持续时间”的不同含义。

**潜在后果**

- 时间误差 10ms 和误差 10 秒在交叉熵中同样是“分类错误”。
- 稀有持续时间难以泛化到相邻时长。
- 模型可能精确 token 准确率低，但事件时间已经接近。
- 位置预测容易形成局部尖锐、多峰分布。
- 3000 个时间类别占词表绝大多数，稀释了结构 token 学习。

这不一定需要立即重做 tokenizer，但应通过诊断确认：

- 时间预测误差的绝对厘秒分布。
- 正确时间 token 的平均排名。
- 错误时间是否集中在目标附近。
- frame 时间 token和 duration token 分别的准确率。

如果大部分时间错误只差 1 到 3 个厘秒，说明分类式时间表示是主要损失瓶颈之一。

---

### 6. 音频和谱面之间没有显式局部或单调对齐偏置

每个谱面 token 都能 cross-attend 到全部 2612 个音频位置：

```python
x = x + self.cross_attn(
    self.cross_attn_ln(x),
    xa,
    key_padding_mask=xa_mask,
)[0]
```

见 `src/model.py:156-162`。

模型不知道：

- 当前 frame 时间戳对应哪一段音频。
- 一个 Note 的属性应该主要查看该 frame 附近。
- 生成时间递增时，音频关注点也通常应向后移动。
- Hold/Slide duration 需要查看局部起点与后续结构。

它只能从绝对音频位置编码和已生成 TS token 中自行学习这种关系。4 层 decoder 可能学会，但数据效率较低。

特别是目标区固定在相对 `12..22s`，模型可能学到“多数目标都在中间区域”，却不一定形成精确的 token 时间到 MERT frame 的映射。

**验证方式**

用 `disable_sdpa()` 获取 cross-attention 权重，按生成 frame 的 TS 检查注意力中心：

- 是否落在对应音频时间附近。
- 是否随 frame 时间单调向后。
- 是否大量固定关注歌曲头、padding 边界或少数位置。
- 不同层是否有明确分工。

当前 SDPA 路径不返回权重，见 `src/model.py:101-124`，但 fallback 路径已有 `qk`，因此可做只读诊断实验。

---

### 7. 仅使用 MERT 最后一层，可能丢失适合节奏任务的中间层信息

MERT 缓存固定读取：

```python
model(input_values=batch).last_hidden_state
```

见 `src/mert_cache.py:157-171`。

很多预训练音频模型中：

- 较浅层更保留局部声学和瞬态。
- 中间层更适合节奏与音色。
- 最后层偏向预训练目标需要的高层语义。

谱面生成未必最适合最后一层。当前冻结缓存一旦选错层，后端 decoder 无法恢复已经被弱化的 onset 信息。

这是高度需要实验证据的问题，不能仅凭理论判断。建议在少量歌曲上缓存若干候选层，训练相同的小模型比较：

- 最后一层。
- 中间层。
- 2 到 4 个层的固定平均。
- 可学习标量加权，但 MERT 仍冻结。

比直接把 MERT 解冻更安全，也更省显存。

---

### 8. decoder 深度偏浅，但不能脱离前述问题单独判断

当前是 4 层、768 维、12 头，约 43.7M 参数，见 `config.yaml:41-48`。

4 层需要同时承担：

- 长序列 self-attention。
- 音频 cross-attention。
- token grammar。
- 时间对齐。
- 谱面规划。
- EOS 决策。

深度可能不足，尤其 cross-attention 到任务抽象之间只有很少的迭代步骤。但当前模型已经很宽，增加层数会直接增加：

- 训练时间。
- 推理时间。
- cross-attention 显存。
- 过拟合风险。

在没有确认模型是否真正使用音频、是否受前缀偏移和终止机制限制之前，直接从 4 层改到 8 或 12 层并不是优先方案。

**判断容量是否不足的标准**

- 5 首歌过拟合时，真值前缀自回归仍不能接近完整复现。
- 训练 loss 长期停滞且不是 EOS/时间 token 单项造成。
- 扩大数据后训练和验证指标都同步欠拟合。
- 注意力已有合理音频对齐，但复杂 Slide/多音符 frame 仍学不会。

只有满足这些证据，增加到 6 层才是合理的下一步。

## 性能与实现问题

### 9. 推理没有 KV cache，长序列解码复杂度非常高

每生成一个 token，都会把完整 token 前缀重新送入 decoder：

```python
dec_input = torch.tensor([tokens], ...)
logits = model.logits(dec_input, audio_features, audio_mask)
```

见 `src/infer.py:120-125`。

虽然音频 embedding 每个窗口只计算一次，但 decoder 的：

- token embedding。
- 所有层 self-attention。
- 所有层 cross-attention query。
- MLP。

都会从头计算。

随着序列长度增长，单步 self-attention 是近似 `O(L²)`，从 1 一直重算到 L，整个窗口累计接近 `O(L³)`。撞到 2048 token 时尤其昂贵。

`model.py` 虽然保留了 `kv_cache` 参数，但当前没有安装 cache hook，也没有调用路径传入 cache。现有实现不能被视为已支持 KV cache。

此外，潜在 cache 实现还有两个风险：

- `src/model.py:211` 用 `next(iter(kv_cache.values()))` 推断位置 offset，若取到 cross-attention cache，长度可能是 2612 而不是文本长度。
- fallback causal mask 使用 `mask[:n_ctx, :n_ctx]`，不支持 query 长度 1、key 长度为历史总长的增量解码，见 `src/model.py:115-119`。

这主要是吞吐问题，不直接解释生成质量，但它限制了大规模整曲验证和采样策略实验。

---

### 10. 训练始终计算到 2047 个 decoder 位置，浪费显存和算力

数据集中所有样本固定 padding 到 2048：

- `src/dataset.py:385-402`
- `src/train.py:260-269`

即使一个 batch 中最长有效序列只有几百 token，模型仍计算完整的 2047 个位置。因果 attention 保证尾部 PAD 不影响前面的有效 token，正确性没有明显问题，但计算量非常浪费。

因为 self-attention 对文本长度是平方复杂度，将长度从 2048 裁到 1024 理论上可把这部分 attention 矩阵缩小到四分之一。

最小优化是 collate 或训练循环按当前 batch 最大有效长度裁切：

```text
tokens[:, :batch_max_length]
mask[:, :batch_max_length]
loss_mask[:, :batch_max_length]
```

这不改变模型架构和训练语义，风险远低于结构改造。

---

### 11. 音频 cross-attention 长度过大且完全不降采样

每层每个 token 都对 2612 个音频位置做 cross-attention。一个完整 batch 的主要注意力规模大致为：

```text
8 batch × 12 heads × 2048 text × 2612 audio
```

单层就约有 5.1 亿个 query-key 关系。SDPA 会避免显式保留完整注意力矩阵，因而不一定直接 OOM，但计算仍然很重。

75Hz 对 10ms 时间戳任务并不算高，实际上音频帧间隔约为 13.3ms；然而整整 34.83 秒保持 75Hz，未必每一层都需要如此细的全局访问。

可以考虑的低风险实验：

- 只对 `0..22s` 或略带右上下文的区域做 cross-attention。
- 两倍时间降采样到约 37.5Hz。
- 保留高分辨率局部特征，同时增加低分辨率全局特征。
- 根据已生成 frame 时间限制 cross-attention 到局部窗口。

但在没有 attention 对齐诊断前，不建议直接降采样，因为 Slide、快速纵连和密集节奏可能确实需要较高时间分辨率。

## 其他潜在问题

### 12. 音频绝对位置编码与 MERT 特征直接相加，缺少尺度控制

音频特征经过 LayerNorm 后大致是单位尺度，位置编码初始化标准差为 `0.02`：

- `src/model.py:233-251`

训练后位置编码没有幅度约束，可能出现：

- 位置编码过弱，模型难以精确定位。
- 位置编码过强，覆盖 MERT 内容。
- 不同位置通过绝对 embedding 被记忆，而非学习可迁移的相对关系。

建议只做诊断，先记录：

- LayerNorm 后 MERT 的 RMS。
- `audio_position` 的 RMS 与最大范数。
- 两者的比例。
- 不同位置编码之间的相似度。

没有证据时不必急着换 RoPE 或相对位置编码。当前窗口语义固定，绝对位置本身是合理选择。

---

### 13. 没有歌曲级难度、风格和密度条件

模型只训练固定 Master，不输入：

- 难度等级。
- 目标音符密度。
- BPM。
- 谱师风格。
- 歌曲全局风格。
- 目标复杂度。

如果数据中的 Master 风格差异很大，同一段音频可能存在多种合理谱面，普通交叉熵会把这种多模态目标压缩成平均偏好。结果可能是：

- 生成谱面保守。
- 密度不稳定。
- 局部音符类型正确，但整体难度不像 Master。
- 对特定谱师风格过拟合。

当前只生成单一难度时，难度 token 不是刚需；但如果后续要混合 Expert/Master 训练，必须先加入显式难度条件，而不能仅靠输出写入哪个 `&inote_*`。

---

### 14. 贪心解码会放大局部错误，但采样不是首要修复

当前每一步都使用 `argmax`：

- `src/infer.py:138`

约束解码保证结构合法，但合法集合中仍可能有多个音乐上合理的候选。贪心搜索的问题包括：

- 一次局部错误无法恢复。
- EOS 和继续生成的决策非常脆弱。
- 时间戳稍微选错后，后续语法状态和资源状态都会改变。
- 跨窗口前缀进一步传播错误。

不过，在模型尚未稳定学会 EOS、音频对齐和真实前缀续写前，不建议直接上 beam search。Beam search 会显著放大当前无 KV cache 的性能问题，也可能偏好更长序列。

合理顺序是：

1. 先验证真值前缀下的 greedy 能否工作。
2. 修复 EOS 与音频对齐问题。
3. 实现 KV cache。
4. 再比较 greedy、短 beam、受控采样。

## 已确认没有明显错误的部分

以下实现从静态检查上看基本合理：

- SDPA 的布尔 `audio_mask` 使用 `True` 表示可参与注意力，和当前 `audio_mask` 语义一致：`src/model.py:101-111`。
- fallback 路径用 `~key_padding_mask` 屏蔽补零帧：`src/model.py:118-123`。
- causal self-attention 在当前非 cache 全序列调用下正确。
- PAD 只在有效序列尾部，因此没有文本 padding mask 也不会污染此前有效 token。
- LayerNorm 强制 float32 计算后转回输入 dtype，有助于半精度稳定：`src/model.py:29-31`。
- 输出投影与 token embedding 权重共享是常见且合理的参数节约方式：`src/model.py:221-224`。
- 音频补零帧由 cross-attention mask 屏蔽，源码自检也覆盖了修改 padding 值不影响 logits：`src/model.py:280-292`。

## 建议的排查顺序

不建议现在同时改 encoder、层数、tokenizer 和解码器。最有效的顺序是：

1. **建立分项诊断**
   统计 frame 时间、Note 类型、lane、duration、Slide 属性和 EOS 的独立 teacher-forcing 指标，不再只看总 token accuracy。

2. **量化 exposure bias**
   对同一批样本比较 teacher forcing、真值前缀自回归、生成前缀自回归三种结果。

3. **确认模型是否使用音频**
   分别对音频做时间打乱、跨歌曲替换、全均值替换，并观察真值前缀生成指标下降幅度。如果影响很小，说明 decoder 主要在做谱面语言模型。

4. **检查 cross-attention 对齐**
   检查生成 frame 时间与注意力峰值是否对应、是否单调移动。

5. **增加最小音频适配器**
   只加残差 MLP 或 1D 局部卷积，保持 MERT 冻结，比较整曲 F1。

6. **单独处理结束机制**
   比较 EOS 权重、显式目标结束条件以及空目标窗口表现，不要把撞上限简单归因于模型容量不足。

7. **正常全量训练再测正则**
   将 `过拟合歌曲数` 改回 0 后，比较无 dropout 和 `0.05` dropout；过拟合实验阶段保持无 dropout。

8. **最后才扩大 decoder**
   只有确认音频被有效利用、终止正常且模型仍欠拟合时，再从 4 层增加到 6 层。

9. **性能优化独立进行**
   先做 batch 动态裁切，再实现正确的 decoder KV cache；二者不应与质量架构实验混在同一个变量中。

**结论**
当前最可能限制生成质量的三项是：**真实前缀训练与递归前缀推理的分布偏移、音频侧缺乏任务适配和时序建模、EOS 缺乏明确的目标区结束表示**。4 层 decoder 可能偏浅，但现有证据不足以证明“加层”是第一优先级；直接扩大模型很可能只是用更多参数吸收前述结构缺陷，并增加过拟合与推理成本。

# 数据-训练-推理全链路问题评估

工程对配置兼容、缓存 generation 和严格规则过滤已有较强防护，但仍需重点验证三个高风险交界面：MERT 帧时间轴是否与固定 75 Hz 假设完全一致、谱面索引是否可能静默漏歌或接受损坏缓存、恢复训练时是否保存并恢复了足够的随机与优化状态。下面继续追踪实际张量和生成路径。


张量链路里已确认两类实际问题：续训只恢复模型/优化器，却重置“历史最优值、早停计数和随机状态”，因此续训不等价于连续训练，并可能立刻覆盖真正的 `best.pt`/`best_gen.pt`；生成最优模型又只按固定排序后的少量验证歌曲挑选，指标方差和选择偏差会很大。还在核对数据审计是否真的覆盖约束器全部规则，以及推理起始时间和尾部边界。

实现与文档存在一处明显脱节：`constrained_decode.py` 顶部仍声称整曲审计“尚未实现”，但 `chart_cache.py` 已调用 `validate_frames()`；更重要的是当前审计对每个 frame 单独检查 token 语法，再用简化记录检查跨帧资源，未证明与完整前缀状态机完全等价，尤其 Slide 的完整签名、属性规范和同 TS 多 frame 场景需要实测覆盖。最后我会运行现有轻量自检，并用缓存/数据统计验证风险是否在真实数据上触发。

发现一条会直接破坏正式推理输出的高严重度缺陷：约束器明确允许 `wait=0, trace>0` 的 Slide，但 `maidata_parser._slide_to_text()` 在这种生成结果上执行 `240.0 / wait_sec`，会 `ZeroDivisionError`，导致模型已生成完谱面却无法写出 `maidata.txt`。这不是理论风险，约束语言主动允许该结构。


**总体结论**
当前工程的主链路设计整体连贯，尤其是以下方面已经做得较好：

- 训练和推理共享窗口定义、MERT 特征提取、token 编码与正式 `overlap_infer()`。
- train/val 按歌曲划分，避免普通模式下窗口级数据泄漏。
- 音频补零位置通过 cross-attention mask 屏蔽。
- token 超长直接拒绝，不静默截断。
- chart cache 使用 generation + 原子指针发布。
- checkpoint 对模型、窗口和 MERT 实现做了严格兼容检查。
- 训练数据先经过整曲硬规则过滤。
- 推理采用完整前缀状态机约束，并在异常中断时丢弃未闭合 token。
- 当前基础自检全部通过。

但全链路仍存在几个重要问题。其中最严重的是：**约束解码可以生成一种 `maidata` 输出器无法处理的合法 token 序列，导致推理在最后写文件阶段崩溃。**

## 严重问题

### 1. 合法约束输出可能让 `maidata` 生成器除零崩溃

位置：

- `src/constrained_decode.py:672`
- `src/constrained_decode.py:673`
- `src/maidata_parser.py:1505`
- `src/maidata_parser.py:1517`

约束器允许 Slide：

```text
wait = TS_0
trace > TS_0
```

具体逻辑是：

```python
traces = POSITIVE_TIMES if wait == 0 else TIMES
```

这明确允许零等待、正滑动时长。

但是 `_slide_to_text()` 在 `trace_duration > 0` 且等待时长不是默认值时执行：

```python
wait_bpm = 240.0 / wait_sec
```

当 `wait_sec == 0` 时直接触发 `ZeroDivisionError`。

我用一个符合约束语言的 Slide 实际调用 `compiler.generate()`，已经稳定复现：

```text
ZeroDivisionError: float division by zero
```

影响：

- 模型完成整曲生成后，`frames_to_maidata()` 可能崩溃。
- 训练的整曲 F1 验证不会发现，因为 `validate_generation()` 只比较 Frame，不调用 `generate()`。
- 当前训练集的 143,766 个 Slide segment 中没有发现这种结构，所以训练数据不会教模型该结构，但约束器会主动向模型开放它。
- 约束器开放但训练中从未出现，反而使这种输出更不可预测。

建议优先级：最高。

最小修复方向：

- 让 `_slide_to_text()` 明确处理 `wait_sec == 0`。
- 修复后增加 `Frame -> generate -> parse` 自检。
- 不建议单纯在约束器中禁止，因为约束规范明确认为 `wait=0, trace>0` 合法，而且 tokenizer 本身能表达。

### 2. 续训会覆盖真正的历史最优模型

位置：

- `src/train.py:204`
- `src/train.py:455`
- `src/train.py:507`
- `src/train.py:529`

恢复 checkpoint 时只恢复：

- model
- optimizer
- scheduler
- scaler
- epoch

但进入主循环后重新初始化：

```python
best_val_loss = float("inf")
best_val_content_f1 = -1.0
bad_epochs = 0
```

因此恢复后的第一轮一定会：

- 把当前模型保存成新的 `best.pt`，即使它比恢复前的历史最优验证损失差。
- 把当前模型保存成新的 `best_gen.pt`，即使生成 F1 更差。
- 把早停耐心重新归零。

这在当前配置中尤其危险：

```yaml
恢复检查点: checkpoints/best.pt
推理检查点: checkpoints/best_gen.pt
```

从 `best.pt` 续训时，第一轮可能同时覆盖 `best.pt` 和 `best_gen.pt`。其中 `best_gen.pt` 原本可能来自完全不同、生成效果更好的 epoch。

建议：

- checkpoint 保存并恢复 `best_val_loss`、`best_val_content_f1`、`bad_epochs`。
- 或恢复时从现有 checkpoint 字段初始化历史最优值。
- 如果从 `best.pt` 恢复，不应默认认为它也是历史生成 F1 最优点。

### 3. 续训不等价于连续训练，随机轨迹没有恢复

位置：

- `src/train.py:60`
- `src/train.py:168`
- `src/train.py:206`

当前没有保存和恢复：

- Python `random` 状态
- NumPy RNG 状态
- PyTorch CPU RNG 状态
- CUDA RNG 状态
- DataLoader shuffle generator 状态

恢复进程启动时又重新执行：

```python
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

结果：

- 恢复后 DataLoader 的 shuffle 从初始随机序列重新开始，而不是接着中断点。
- 在线音频增强使用 `torch.rand()`、`torch.randn_like()`、`torch.randint()`，其增强序列也会重置。
- 多 worker 情况下每个 worker 的随机状态同样无法续接。
- 同一 checkpoint 反复恢复会重复相似的数据顺序与增强轨迹。

这不一定导致模型错误，但会破坏“严格续训”和可复现实验的语义。当前 `checkpoint_config()` 的注释称其用于“严格续训”，实际还达不到严格续训。

## 高风险问题

### 4. `best_gen.pt` 使用固定、很小的验证歌曲子集选模

位置：

- `src/train.py:164`
- `src/train.py:166`
- `src/train.py:478`
- `config.yaml:81`

普通训练默认：

```yaml
整曲验证歌曲数: 2
```

选取方式是：

```python
gen_val_paths = sorted(val_charts)[:VAL_GEN_CHARTS]
```

这意味着每轮都只在按 MERT 缓存哈希路径排序后的固定两首歌曲上选择 `best_gen.pt`。

问题：

- 两首歌的方差很大。
- 不是由 seed 随机选取，也不轮换。
- 排序依据实质上是 MD5 派生的缓存文件名，不具有数据代表性。
- checkpoint 会过拟合这两首歌的内容分布、长度和 note 类型。
- 只要其中一首特别简单或特别复杂，F1 选择就会明显偏斜。
- 对小验证集进行 500 轮反复挑选，本身也会形成验证集选择过拟合。

建议：

- 至少固定 seed 后从验证歌曲中抽取更大的代表性子集。
- 更稳妥的是每轮轮换子集，定期对全部验证歌曲评估。
- `best_gen.pt` 最好使用固定完整验证集或足够大的固定评估集。
- 同时输出按歌曲 macro F1，避免长谱面完全主导 micro F1。

### 5. 内容指标的一对一匹配算法不是最大匹配，可能低估 TP

位置：

- `src/content_metrics.py:15`

当前逻辑按预测顺序逐个寻找最近的未使用目标：

```python
for pred_time, pred_type in pred_events:
    ...
    if dt <= best_dt:
        best_i = i
```

这是局部贪心，不保证得到容差窗口内的最大一对一匹配数。

已构造可复现反例：

```python
pred = [(0.010, A), (0.020, A)]
truth = [(0.000, A), (0.019, A)]
tolerance = 0.011
```

当前算法得到：

```text
TP = 1
```

但正确最大匹配可以得到：

```text
0.010 -> 0.000
0.020 -> 0.019
TP = 2
```

影响：

- precision、recall、F1 可能被低估。
- 相同类型音符密集出现时最明显。
- `best_gen.pt` 可能因匹配顺序而选错 epoch。
- 预测事件顺序发生细微变化时，F1 可能非平滑跳变。

由于事件已按时间排序，通常不需要复杂图算法。可以按 note signature 分组后使用有序双指针或动态规划完成最大匹配。

### 6. MERT 时间轴全链路硬编码为精确 75 Hz，但实际模型一秒输出 74 帧

位置：

- `src/mert_cache.py:361`
- `src/config.py:232`
- `src/chart_cache.py:20`
- `src/dataset.py:216`
- `src/infer.py:25`

MERT 自检明确断言：

```python
_output_length(model, processor.sampling_rate) == 74
```

但后续所有时间换算均使用：

```python
frames_per_sec = 75
```

这未必立即构成错误，因为长输入的卷积输出帧间距可能确实接近 75 Hz，单独一秒得到 74 帧主要来自卷积有效边界。但当前工程没有把这个关键假设写成可验证的不变量。

潜在问题：

- `mel.shape[0] / 75` 估算的歌曲长度可能与实际音频时长有固定边界误差。
- `round(time * 75)` 与真实特征中心位置可能存在几毫秒偏差。
- 分块裁剪后按 feature center 保留核心区，帧数未必严格等于 `round(duration * 75)`。
- 训练和推理彼此一致，所以系统可能“自洽地偏移”，但相对真实音频节拍仍可能有系统误差。

当前缓存长度范围为 5,155 到 19,612 帧，没有直接暴露异常，但缺少以下验证：

- 原始音频时长与最后一个 MERT feature center 的差异。
- 分块提取与整曲一次提取的时间轴/特征差异。
- 若干歌曲上 `feature_index / 75` 与真实 center 的最大偏差。

建议先测量，不建议未经测量就大改窗口定义。

### 7. chart cache 只依赖 `.npy` 文件状态，没有验证 MERT 元数据

位置：

- `src/chart_cache.py:100`
- `src/chart_cache.py:113`
- `src/chart_cache.py:119`

chart index 的 source manifest 记录：

```python
"mel_state": _state(mel_path)
```

但没有记录或验证同名 `.json` 元数据。

正常情况下 `ensure_chart_cache(build_mel=True)` 会先调用 `rebuild_mert_cache()`，因此主训练入口通常安全。但仍有风险：

- `valid_pairs` 非空时，`ChartDataset` 使用 `build_mel=False`。
- `.npy` 被手工替换但碰巧维持 size/mtime 时不会失效。
- `.json` 缺失或损坏时，现有 chart generation 仍可能被复用。
- chart index 只检查 shape，不检查特征是否有限值。
- `np.load()` 成功但内容包含 NaN/Inf 时，会直接污染训练。

更稳妥的最小方案：

- chart manifest 同时记录 MERT `.json` 的状态或特征签名摘要。
- Dataset 首次 mmap 时至少检查 `np.isfinite`，或者缓存写入时记录并验证有限值。
- 缓存读取应检查 dtype、shape 和预期元数据，而不只是文件存在。

## 中等风险问题

### 8. 缺失音频的 32 份谱面被静默排除

真实数据统计：

```text
maidata.txt: 1566
完整 track + MERT cache: 1534
缺少 track.mp3: 32
```

位置：

- `src/mert_cache.py:266`
- `src/chart_cache.py:104`
- `src/chart_cache.py:109`

两处扫描都直接跳过缺少音频的目录，没有最终汇总这些目录。

这不会污染训练，但会造成数据规模误判。用户看到的是“扫描了 charts”，实际有 32 首完全未进入数据链路。

建议启动时输出：

- maidata 总数
- 缺少 track 的数量和路径
- 缺少/损坏 MERT 的数量
- 缺少目标 level 的数量
- 严格规则排除数量
- 最终进入训练的歌曲数量

### 9. MERT 构建失败后训练可以继续使用不完整数据集

位置：

- `src/mert_cache.py:317`
- `src/mert_cache.py:339`
- `src/chart_cache.py:247`

MERT 构建对单首失败采用：

```python
跳过
```

然后 chart cache 只扫描存在 `.npy` 的歌曲。

这适合批量容错，但对训练任务来说可能过于静默：

- 一批音频因系统性 FFmpeg/TorchCodec 问题失败时，训练仍会开始。
- 数据集规模可能骤减。
- 如果失败与音频编码、版本或歌曲年代有关，会形成非随机数据偏差。
- `rebuild_mert_cache()` 的 `(count, skipped)` 返回值没有被 `ensure_chart_cache()` 检查。

建议：

- 有失败时明确打印失败比例。
- 超过阈值时终止训练。
- 至少把失败路径写进 chart manifest，避免只看到“最终可用歌曲数”。

### 10. 数据审计和推理状态机不是同一条完整执行路径

位置：

- `src/constrained_decode.py:431`
- `src/constrained_decode.py:442`
- `src/constrained_decode.py:458`

`validate_frames()` 对每个 frame 单独做：

```python
[SOS, encode_frame(frame), EOS]
```

的 token 结构验证，然后使用 `_ReplayState + _add_frame_records()` 检查跨帧资源。

而推理约束器使用完整 token 前缀 `_PrefixReplay`。

这种拆分容易产生语义漂移：

- frame 内 token canonical 结构由 `_PrefixReplay` 检查。
- 跨 frame 只保留 `_NoteRecord` 的简化状态。
- Slide 的跨 frame 表示被压缩为 press lane，不保留完整路线信息。
- 同 TS 多个原始 frame 的处理依赖 `_ReplayState`，没有逐 token 走完整前缀。
- 修改 `_PrefixReplay` 时容易忘记同步 `_add_frame_records()`。

我尝试对全库执行“通过 `validate_frames()` 的整曲再完整重放 `allowed_tokens()`”差分扫描，但纯 Python 完整前缀重放是近似二次复杂度，10 分钟内没有完成。这本身也说明当前缺少一个高效、可持续运行的等价性回归检查。

建议：

- 审计时按整曲 token 顺序复用同一个增量状态机。
- 或至少在缓存构建时对每个滑窗完整 token 做一次 allowed-token 重放。
- 增加抽样差分检查，覆盖同 TS 多 frame、跨窗口 Hold、Touch Hold 和多段 Slide。

### 11. 约束器文档与当前实现已经不一致

位置：

- `src/constrained_decode.py:284`
- `src/chart_cache.py:177`

文档仍写：

> 尚未实现的是 chart_cache 构建前的整曲质量审计和自动排除报告

但 `chart_cache.py` 已经调用：

```python
violations = validate_frames(level.frames)
```

并整曲排除。

风险不是运行时错误，而是后续维护者可能依据错误文档：

- 重复实现已有功能。
- 误以为旧缓存仍未经审计。
- 不清楚当前 `CACHE_VERSION=8` 已经包含哪些规则语义。

### 12. 解析器会静默归一化或忽略部分源谱内容，缺少数据质量计数

真实全库扫描时出现了多种日志：

- 伪 EACH 合并
- EACH 空分支忽略
- 星形 TAP 归一化
- 伪 HOLD 归一化
- TAP 后孤立时长忽略
- 多余花括号忽略

这些行为可能是合理的清洗，但目前只打印一次日志，不进入 manifest 统计。

影响：

- 无法知道有多少歌曲、多少事件被归一化。
- `maidata.txt` 变化后，某种新异常可能大规模出现，但训练仍正常开始。
- 无法区分“正常语义规范化”和“容错丢失源信息”。

建议把 parser warning 结构化计数，并在 chart manifest 中保存每类规则的歌曲数和事件数。

### 13. 音频增强的噪声尺度按整个窗口所有特征元素的单个标准差计算

位置：

- `src/dataset.py:188`

```python
valid.std(unbiased=False)
```

这是对时间和 768 个特征维度一起求一个标量标准差。

问题：

- MERT 不同通道的尺度可能不同。
- 单个高方差通道会抬高所有通道噪声。
- 后续模型有窗口级 LayerNorm，实际噪声作用可能比配置直觉更弱或更怪。
- 当前 `feature_noise_std=0`，所以暂时不触发。

如果未来启用，应先比较：

- 每通道标准差噪声
- 全局标准差噪声
- LayerNorm 前后的实际扰动幅度

时间遮挡本身实现较清楚，并且只遮真实帧。

### 14. `start_sec` 不是严格裁剪边界

位置：

- `src/infer.py:232`
- `src/infer.py:240`

当 `start_sec > 0` 时：

- `committed` 初始为空。
- 第一窗没有开始时间之前的真实谱面前缀。
- 模型只依赖音频上下文，从空 token 前缀开始生成。
- 输出文件从 `start_sec` 后开始，没有说明这是“局部生成”而不是完整谱面。

这不是算法 bug，但容易被误解为“从中间继续生成”。它实际上无法恢复此前 Hold/Slide/节奏 token 上下文。

建议 CLI 文档明确：

- `--start-sec` 是无历史谱面条件下的局部起点。
- 不等价于从已有谱面继续生成。
- 若要续写，需要接受已有 maidata 并构造 committed prefix。

### 15. 训练损失的 token 平均方式会偏向高密度窗口

位置：

- `src/train.py:272`
- `src/train.py:282`

损失是所有受监督 token 的 micro average。token 多的窗口权重更高：

- 多 Note、多段 Slide 的窗口贡献更大。
- 空目标窗口基本只贡献 EOS。
- 简单节奏和稀疏窗口贡献较小。

这可能是合理选择，但需要意识到：

- “每窗口表现”与“每 token 表现”不是同一目标。
- EOS 额外权重进一步改变了 loss 的标度。
- train/val loss 不能直接解释为每个窗口的平均质量。

建议保留当前 loss，同时额外记录：

- 每窗口平均 loss
- 空目标/非空目标窗口 loss
- 按 token 类型的 loss 或 accuracy

## 低风险及维护问题

### 16. `mask` 在训练中读取但未使用

位置：

- `src/train.py:257`

```python
mask = batch["mask"].to(...)
```

之后没有使用。真正监督范围由 `loss_mask` 控制，因此功能无误，但变量会误导读者以为 decoder 有文本 padding attention mask。

当前 decoder 使用 causal mask，没有 key padding mask；由于 loss 不监督 padding 后的位置，且 causal attention 不会让有效位置看到未来 PAD，所以训练语义仍然成立。

可以删除这次无用传输，或明确注释 `mask` 只供验证前缀长度使用。

### 17. 推理帮助文本仍引用旧 checkpoint 路径示例

位置：

- `src/infer.py:4`

文档写：

```text
--checkpoint ckpts/best.pt
```

实际配置目录为：

```text
checkpoints/best_gen.pt
```

只是文档陈旧。

### 18. `validate_dataset()` 已经不是正式数据入口，默认缓存目录也不同

位置：

- `src/dataset.py:20`
- `src/dataset.py:28`

它默认使用：

```python
charts_dir.parent / ".cache" / "charts"
```

正式缓存则是 `.cache/mert`。当前主训练不调用它，但保留这个函数容易让后续诊断得到“全部缺少 mel cache”的错误结论。

建议删除未使用旧入口，或者改为复用 `CONFIG.paths.mert_cache_dir` 和正式 cache key。

### 19. 配置默认处于过拟合和自动恢复模式

位置：

- `config.yaml:86`
- `config.yaml:88`

当前为：

```yaml
过拟合歌曲数: 5
恢复检查点: checkpoints/best.pt
```

因此直接运行 `uv run src/train.py` 并不是正常全量训练，而是：

- 只选 5 首歌。
- 训练和验证使用同一批歌曲。
- 自动从 `best.pt` 恢复。

如果这是当前排查阶段的刻意设置，没有问题；但它是非常危险的默认值。后续忘记改配置，可能误以为在做正式实验。

正式默认建议：

```yaml
过拟合歌曲数: 0
恢复检查点: null
```

过拟合排查再显式修改。

## 已验证正常的部分

以下自检均通过：

- `src/config.py`
- `src/tokenizer.py`
- `src/model.py`
- `src/constrained_decode.py`
- `src/content_metrics.py`
- `src/infer.py --help`

代码审查确认以下语义目前一致：

- 训练窗口与推理窗口都使用 `window_start = target_start - 12s`。
- 都使用 `round(time * 75)` 截取逻辑音频窗口。
- 都使用厘秒量化后判断 `6..12s` 前缀和 `12..22s` 目标。
- 训练 supervision 从第一个目标 frame 的 `FRAME_START` 开始，到 EOS 结束。
- 没有目标 frame 时只监督 EOS。
- 推理只提交目标区对应的 10 秒。
- 后续窗口前缀来自已提交结果的最后 6 秒。
- 音频 padding mask 在 SDPA 和 fallback attention 两条路径上都有效。
- 旋转仅改变 lane/touch token，不改变音频和 loss mask。
- 普通模式按歌曲划分，不存在同一首歌窗口跨 train/val。
- `uint16` 足以容纳当前 `0..3071` 词表。
- token 长度超过 2048 时拒绝训练，不会截断闭合结构。
- MERT cache 会随模型文件、库版本、Python 版本、分块参数和源音频元数据失效。

## 建议修复顺序

1. 修复零等待 Slide 的 `maidata` 输出崩溃，并添加 round-trip 自检。
2. 修复续训的历史最优值、早停状态和 checkpoint 覆盖问题。
3. 保存和恢复 RNG 状态，明确“严格续训”语义。
4. 修复内容指标的最大一对一匹配。
5. 扩大或轮换生成验证歌曲，降低 `best_gen.pt` 选择偏差。
6. 给数据入口增加完整统计，包括 32 份缺音频谱面和 MERT 构建失败。
7. 验证 MERT 75 Hz 时间轴假设，先测量真实 feature center 误差。
8. 统一 `validate_frames()` 和完整推理状态机，建立可运行的差分回归检查。
9. 清理过期文档、旧 `validate_dataset()` 和危险默认配置。



