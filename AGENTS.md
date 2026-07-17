# 项目说明

- 添加依赖使用 `uv add <包名>`，运行脚本使用 `uv run <文件名>.py`。
- 注释和日志使用中文。
- 用户没有要求使用子代理时，不要使用子代理。
- 轻量回归断言写在源码 `_self_check()` 中，根目录 `test.sh` 统一运行。
- 正式推理为每首歌创建独立目录，目录内只能有 `maidata.txt` 和 `track.mp3`。
- **不要轻易改动**检查点的兼容性，训练一次模型要很久。

## 当前架构

- `config.yaml`：全部可调默认参数。
- `src/config.py`：配置读取和校验。
- `src/chart.py`：纯谱面数据对象。
- `src/maidata_parser.py`：只提供 `parse_maidata(text)` 和 `generate_maidata(chart)`。
- `src/dataset.py`：索引全量歌曲并按歌曲稳定划分训练、验证和测试集；每轮现场解码、增强并提取 log-mel，不保存特征缓存。
- `src/model.py`：以浮点难度为条件的 BERT Encoder 逐帧模型，分别预测 Tap 0/1/2 数量、Hold 0/1/2 数量和两个 Hold 时长。
- `src/train.py`：使用 1024 帧窗口、512 帧步长进行全量多歌曲训练，记录窗口损失和多容差整曲 F1，并支持完整断点恢复。
- `src/infer.py`：唯一正式推理入口，以 1024 帧窗口、512 帧步长重叠推理并加权合并整曲结果。
- `src/constrained_decode.py`：旧 token 严格解码规范，只保留说明，没有可执行实现。

训练使用歌曲中可用的 Basic 至 Re:Master 谱面；Diving-Fish 匹配失败时使用谱面字符串
难度近似值。正式推理默认使用 `level_idx=5`，对应 Master。旧 checkpoint、特征缓存、
MERT、tokenizer、CNN、自回归模型和 v2 检查点均不兼容当前架构。

训练和推理不接受命令行参数或环境变量覆盖。先修改 `config.yaml`，再运行
`uv run src/train.py` 或 `uv run src/infer.py`。
`训练.训练歌曲数` 为 `0` 时使用完整训练集，正整数时使用稳定划分后的前 N 首歌曲。
