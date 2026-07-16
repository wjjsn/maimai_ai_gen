# 项目说明

- 添加依赖使用 `uv add <包名>`，运行脚本使用 `uv run <文件名>.py`。
- 注释和日志使用中文。
- 不要使用子代理
- 轻量回归断言写在源码 `_self_check()` 中，根目录 `test.sh` 统一运行。
- 正式推理为每首歌创建独立目录，目录内只能有 `maidata.txt` 和 `track.mp3`。

## 当前架构

- `config.yaml`：全部可调默认参数。
- `src/config.py`：配置读取和校验。
- `src/chart.py`：纯谱面数据对象。
- `src/maidata_parser.py`：只提供 `parse_maidata(text)` 和 `generate_maidata(chart)`。
- `src/dataset.py`：解码音频并提取 log-mel，加载单曲多难度谱面，生成四列逐帧监督轴和滑窗样本。
- `src/model.py`：以浮点难度为条件的 BERT Encoder 逐帧模型，分别预测 Tap 数量、Hold 数量和两个 Hold 时长。
- `src/train.py`：使用 1024 帧窗口、10 帧步长进行单曲多难度过拟合训练，并执行整曲生成评估。
- `src/infer.py`：唯一正式推理入口，以 1024 帧窗口、512 帧步长重叠推理并加权合并整曲结果。
- `src/constrained_decode.py`：旧 token 严格解码规范，只保留说明，没有可执行实现。

训练使用歌曲中可用的 Basic 至 Re:Master 谱面；正式推理默认使用 `level_idx=5`，对应
Master。旧 checkpoint、特征缓存、MERT、tokenizer、CNN 和自回归模型均不兼容当前架构。

训练和推理不接受命令行参数或环境变量覆盖。先修改 `config.yaml`，再运行
`uv run src/train.py` 或 `uv run src/infer.py`。
