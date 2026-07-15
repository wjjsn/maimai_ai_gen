# 项目说明

- 添加依赖使用 `uv add <包名>`，运行脚本使用 `uv run <文件名>.py`。
- 注释和日志使用中文。
- 用户没有要求用，就不要使用子代理
- 轻量回归断言写在源码 `_self_check()` 中，根目录 `test.sh` 统一运行。
- 推理时为每首歌创建独立目录，目录内只能有 `maidata.txt` 和 `track.mp3`。

## 当前架构

输入歌曲音频，训练时由 DataLoader 现场提取 200Hz log-mel，不读写任何特征或谱面缓存。
整曲 CNN 等长预测一条短音轴和两条持续轴，再转换为 `maidata.txt`。

- `config.yaml`：全部可调默认参数。
- `src/config.py`：配置读取和校验。
- `src/audio_features.py`：现场解码音频并提取 log-mel。
- `src/chart.py`：纯谱面数据对象。
- `src/maidata_parser.py`：只提供 `parse_maidata(text)` 和 `generate_maidata(chart)`。
- `src/dataset.py`：发现歌曲、现场加载特征、生成监督轴。
- `src/model.py`：整曲 CNN。
- `src/train.py`：训练、验证和最终整曲生成。
- `src/infer.py`：唯一正式推理入口。
- `src/constrained_decode.py`：旧 token 严格解码规范，只保留说明，没有可执行实现。

训练和推理默认使用 `level_idx=5`，对应 Master。旧 checkpoint、缓存、MERT、tokenizer、
滑窗和自回归模型均不兼容当前架构。

训练和推理不接受命令行参数或环境变量覆盖。先修改 `config.yaml`，再运行
`uv run src/train.py` 或 `uv run src/infer.py`。
