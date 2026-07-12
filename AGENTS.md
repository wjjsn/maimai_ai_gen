# 项目说明

- **添加依赖**：使用 `uv add <包名>`，它会自动创建虚拟环境并更新 `pyproject.toml`。
- **运行脚本**：使用 `uv run <文件名>.py`，它会自动激活并管理环境。
- **注释用中文，日志用中文**
- **测试风格**：不要创建独立的 `test_*.py` 测试文件。将轻量回归断言写入对应源码文件的 `_self_check()`，并在该文件作为主程序执行时运行，例如 `uv run src/tokenizer.py`。

## 项目全览

本项目是一个 maimai 谱面生成实验工程：输入歌曲音频，训练 Whisper 风格的 encoder-decoder 模型，自回归生成 simai/maidata 谱面 token，再还原成 `maidata.txt`。

核心目录：

- `charts/`：训练数据目录。每首歌通常包含 `maidata.txt` 和 `track.mp3`。
- `.cache/charts/`：log-mel 缓存及同名 `.json` 元数据。`track.mp3` 的大小、修改时间或 Mel 参数变化时会自动重建对应文件。
- `.cache/chart_index/`：持久化的滑窗谱面索引。按难度、Mel 参数、滑窗步长和目标区配置分目录保存 generation；`current.json` 原子切换当前 generation。
- `checkpoints/`：训练或过拟合保存的模型权重。
- `tmp/`：推理、过拟合、文本对比等临时输出。
- `doc/history.md`：历史排查记录，包含已废弃方案和当时的排查过程；不能单独作为当前行为依据，先以本文件和源码为准。
- `doc/token化设计.md`：token 设计文档，理解谱面 token 格式时看它。
- `doc/Notations of simai.md`：simai 记谱语法参考。

核心代码：

- `src/train.py`：主训练和整曲过拟合的统一入口。负责读取 `ChartDataset`、按歌曲划分 train/val、训练模型、teacher-forcing 验证、调用正式整曲推理计算事件 F1，并保存 `best.pt` 和 `best_gen.pt`。设置 `MAIMAI_OVERFIT_CHARTS=N` 可只选 N 首完整歌曲训练和验证，但窗口、loss、推理、指标都与全量流程一致。训练集会经过 `RotatedDataset` 做 8 方向旋转增强，验证集不增强。
- `src/dataset.py`：数据集读取、滑窗标签语义和旋转增强。输入为 `3000` mel 帧，约 34.83 秒；窗口每 1 秒平移。运行时 mmap 读取 `chart_index` 的 token 与窗口坐标，按需读取 Mel；不再在训练启动时重新解析全量谱面。decoder 可见相对 `6..12s` 的谱面 token 前缀，只对相对 `12..22s` 的中间 10 秒 token 和 EOS 计算 loss，音频左右上下文约 12 秒。验证索引使用 10 秒步进。歌曲头尾通过左右补零保持目标区位置固定，不夹动窗口。
- `src/infer.py`：唯一正式推理入口。每次生成并提交 10 秒，使用前一窗口已生成结果中最后 6 秒的 token 作为当前 decoder 前缀；音频窗口与训练一样把目标固定在相对 `12..22s`。训练中的整曲生成验证直接复用 `overlap_infer()`，不再有 reference/fixed 特殊推理路径。
- `src/maidata_parser.py`：simai/maidata 与内部 token 之间的转换器，文件很长。重点看这些区域：数据结构和 token 常量在文件前部；`parse()` 负责读 `&inote_*`；`_ts_token()` 负责秒到 TS token，当前会 clamp 到 `0..29.99s`；`to_tensor()` 是旧动态分段逻辑，当前主训练已不依赖它构造滑窗；`_parse_token_segment()` 和 `parse_from_tensor()` 负责把生成 token 还原成 frame；`generate()` 负责输出 maidata 文本。普通训练问题通常不用通读整个文件。
- `src/constrained_decode.py`：语法约束解码器。`allowed_tokens()` 根据当前 token 前缀返回下一步允许 token，避免生成非法结构。它只保证语法合法，不保证音乐性。
- `src/model.py`：Whisper 风格模型定义，包括音频 encoder 和文本 decoder。普通训练调参通常不用改它；如果改 mel 输入长度或 `n_audio_ctx`，必须看这里的 encoder positional embedding 长度约束。
- `src/mel_cache.py`：把 `track.mp3` 转成 log-mel 并缓存；以源音频 `size + mtime_ns` 和 Mel 参数自动失效。旧 `.npy` 首次使用只补写元数据，不会全量重算。
- `src/chart_cache.py`：编译并缓存滑窗 token、loss 起点和音频切片坐标。`ChartDataset` 自动调用 `ensure_chart_cache()`；源谱面或 Mel 的 `size + mtime_ns`、难度或窗口配置变化时自动重建。构建使用锁和临时 generation，写完后原子更新 `current.json`，不能删除正在被 mmap 使用的旧 generation。
- `src/check_rotation.py`：旋转增强自检脚本，检查旋转 0/8 次不变、连续 8 次还原、Touch C 不变、旋转后 token 可解析。

难度编号：

- 训练和推理默认都应该使用 `level_idx=5`，对应 `&inote_5` / `&lv_5`，即 Master。
- `level_idx=4` 对应 Expert。
- 如果训练 checkpoint 是用 Expert 训练出来的，只在推理时改 `--level 5` 不会让模型变成 Master，只把推理输出写到`level 5`。

容易踩坑：

- 训练和推理的滑窗定义必须保持一致。改 `PREFIX_START_SEC`、`TARGET_START_SEC`、`TARGET_END_SEC`、`mel_frames`、`n_audio_ctx` 或步长时，要同时检查 `chart_cache.py`、`dataset.py`、`infer.py` 和 `train.py`；同时提升 `CACHE_VERSION`，避免复用语义已变化的 token 缓存。
- 当前 `3000 mel frames` 实际约 34.83 秒，不是严格 30 秒；而 TS token 只能表达 `0..29.99s`。这是后续改进重点。
- 8 方向旋转增强只旋转 token，不改音频。它覆盖所有整体旋转；逆时针 1 次等价于顺时针 7 次。
- `constrained_decode.py` 只做语法约束，不能替代模型学会音乐内容。
- `tmp/` 和 `checkpoints/` 里的结果可能来自旧难度或旧方案，使用前要确认命令参数和 checkpoint 来源。
- 谱面索引中的 token 以 `uint16` 保存，读取后转为 `torch.int64`；词表当前为 `0..3071`。修改词表范围、token 常量、编码或解析语义时，必须提升 `CACHE_VERSION`；若范围不再适合 `uint16`，还要升级缓存格式和存储 dtype。
- 索引用 `loss_start` 表示连续监督后缀：从第一个目标 frame 的 `FRAME_START` 到 EOS；没有目标 frame 时只监督 EOS。修改 token 窗口构造时必须验证它与逐 token `loss_mask` 等价。
- `ChartDataset` 每个 worker 最多保留 8 首歌的 Mel mmap。训练 loader 默认 `MAIMAI_NUM_WORKERS=2`、`MAIMAI_PREFETCH_FACTOR=2`、`MAIMAI_PIN_MEMORY=1`；内存紧张时优先调低这些参数。

## 当前设计问答和结论

### 澄清当前的设计模式

问：当前训练集是不是固定扩大了 8 倍？

答：是。当前训练集用 `RotatedDataset` 固定扩大 8 倍，只包装训练集，验证集不增强。只做 8 轨整体旋转：顺时针 `0..7` 次已经覆盖所有圆周旋转。逆时针 1 次等价于顺时针 7 次，不会增加新样本。如果只做整体旋转，最多就是 8 倍。

问：当前滑窗训练是怎么组装 token 张量的？

答：训练窗口按目标绝对起点 `0, 1, 2...` 每秒平移，逻辑音频窗口起点是 `target_start - 12s`。输入固定为 `3000` mel 帧，左音频上下文为 12 秒、右音频上下文约 12.83 秒；token 包含相对 `6..12s` 的真实谱面前缀和相对 `12..22s` 的目标。训练仍用 `tokens[:, :-1]` 预测 `tokens[:, 1:]`，但独立的 `loss_mask` 只监督目标区完整 frame token 与 EOS，前缀只作为 decoder 上下文。验证索引按 10 秒步进；歌曲头尾补零而不移动窗口，因此目标区始终固定在相对 `12..22s`。

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

问：`3000 mel frames ≈ 34.83s` 是不是当前滑窗方案的严重问题？

答：对当前实现不是。超过提交区的右侧部分只是上下文，不会被提交，也不会进入训练目标。窗口起点固定为 `target_start - 12s`，歌曲尾部通过右补零处理，不会再夹动窗口；目标时间戳始终位于相对 `12..22s`，不会因尾窗超过 `29.99s`。不要武断优先大改窗口长度；应先依据整曲评估确认边界质量是否是瓶颈。

问：要不要改成 `10s 左上下文 + 10s 提交区 + 10s 右上下文`？

答：这是合理方案，能让提交区更居中，边界更稳。如果保留当前 `3000 mel frames`，实际会变成 `10s 左上下文 + 10s 提交区 + 约 14.83s 右上下文`。代价是窗口数和推理次数大约翻倍；再叠加 8 倍旋转，训练成本会明显增加。如果边界质量差，可以考虑改；但应该和整曲评估一起看，不要盲目增加计算量。

问：左边音频上下文里，模型能不能看到已经生成过的 token？

答：能看到其中最后 6 秒。训练 token 包含相对 `6..12s` 的真实谱面前缀，但只对相对 `12..22s` 的目标区计算 loss；推理时会把前一窗口已经提交的最后 6 秒 frame 重新编码为当前窗口前缀。首窗才从 `SOS` 开始。前缀会带来跨窗口连续性，也意味着推理中的前缀错误可能传递到后续窗口；这与训练使用真实前缀的分布存在差异。

问：滑窗要不要改成按 1 秒步进，让同一段 30 秒音频用不同 offset 训练很多遍？

答：当前训练已经按 1 秒步进，让同一段音频以不同相对时间出现在多个窗口中，因此已经覆盖时间平移等变性的主要收益；再额外加密步长只会显著增加高度重复的样本。训练窗口从 `target_start=0,1,2...` 构造，正式推理固定每次提交 10 秒。只有整曲评估证明固定相位或边界仍是瓶颈时，才考虑改变训练采样策略。

问：当前约束解码和要求大语言模型格式化输出 JSON 是同类方法吗？

答：是同一类思想，都是 grammar-constrained decoding。`constrained_decode.py` 的 `allowed_tokens()` 在推理时每一步先拿模型 logits，再把非法 token 置为 `-inf`，只在合法集合里选下一个 token。这只能保证 token 结构合法，不能保证谱面跟音乐对齐、难度合理或有音乐性。如果 parser 一直报警告，优先检查 `allowed_tokens()` 和 `_parse_token_segment()` 是否状态机不一致。

问：训练时能不能也用 `allowed_tokens()` 约束着训？

答：可以，但当前主训练 loss 没有这么做。主训练是普通 teacher forcing：完整词表上预测下一个 token。`allowed_tokens()` 目前主要用于推理、验证里的 greedy decode、过拟合脚本的生成路径。如果要做约束训练，先验证所有训练 target 在对应 prefix 下都属于 allowed set，否则约束器会把正确答案屏蔽掉。建议先加可选开关，例如 `MAIMAI_GRAMMAR_MASK=1`，不要默认启用。朴素实现需要对 batch 里每个位置跑 Python 状态机，可能很慢。


## 当前设计问题的改进

问：当前损失函数能不能正确评估训练集和验证集的效果？

原回答：当前 `CrossEntropyLoss(ignore_index=PAD, label_smoothing=...)` 能正确评估“精确 token 序列预测”的 train/val loss，但不能正确反映“时间对了、音符类型对了、路数无所谓”的谱面效果。`val_acc`、`greedy_accuracy` 都要求 token 完全一致，原来的 `content_match_counts()` 也使用 `repr(note)`，会把 lane、Touch 区域和 Slide 起终点不同判成错误。现有 loss 可以保留为精确 token 复现的辅助指标，但不能单独用来判断生成内容是否正确。

解决方案：已新增 `src/content_metrics.py`，将生成 token 和目标 token 解析为音符事件，进行一对一匹配并计算 precision、recall、F1，`src/train.py` 已改为使用该指标。匹配忽略 Tap/Hold 的 lane、Touch 的具体区域、Slide 的起点/终点/GrandV 中间 lane，以及普通音符的 Break、EX 和 Slide 的 `isSlideBreak`。所有音符必须在 `10ms` 时间容差内且 `NoteType` 一致；Hold 和 Touch Hold 还要匹配持续时间；Slide 还要逐段匹配 `SlideShape`、等待时长、滑动时长、`isClockwise`、`isForceStar`、`isFakeRotate` 和 `isSlideNoHead`。持续时间按 token 的 `10ms` 精度比较，重复预测不能重复命中同一个目标事件。

问：为什么整曲生成几乎每个 10 秒窗口都达到 2048 token，却不生成 EOS？

原回答：不能直接归因于模型生成 slop。需要先检查 EOS 是否进入 loss、是否被 grammar 屏蔽、是否在 teacher forcing 下能预测，以及真值 token 前缀和模型递推前缀下的终止行为是否不同。

解决方案：已对 3 首过拟合歌曲的 318 个训练窗口逐 token 重放 `allowed_tokens()`。64650 个受监督 token 中没有 grammar 违规，318 个 EOS 全部合法且都紧跟 `FRAME_END`，排除了标签错位和约束器屏蔽。统计发现 EOS 只占受监督 token 的 `0.492%`；在 `FRAME_END` 后的二选一决策中，继续 `FRAME_START` 有 8173 次，EOS 只有 318 次，结束率仅 `3.75%`。普通交叉熵下，训练早期始终选择继续下一帧是明显的类别失衡局部最优。当前默认给 EOS 使用 `5x` loss 权重，可通过 `MAIMAI_EOS_LOSS_WEIGHT` 调整。验证日志已新增 EOS 准确率、平均概率、平均排名、误判为 `FRAME_START` 的比例；另用少量真值前缀窗口与正式递推整曲推理对照，并汇总两种模式的撞上限比例和平均新增 token 数。若真值前缀也撞上限，问题在 EOS 学习；若只有正式递推撞上限，问题在模型生成前缀造成的误差传播。
