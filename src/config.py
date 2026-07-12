from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    charts_dir: Path
    mert_cache_dir: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class MertConfig:
    model_dir: Path
    chunk_sec: float
    context_sec: float
    cache_float16: bool
    audio_workers: int
    batch_size: int


@dataclass(frozen=True)
class WindowConfig:
    mert_frames: int
    prefix_start_sec: float
    target_start_sec: float
    target_end_sec: float
    train_stride_sec: float
    infer_stride_sec: float


@dataclass(frozen=True)
class ModelConfig:
    state: int
    head: int
    layer: int
    max_tokens: int


@dataclass(frozen=True)
class TrainingConfig:
    level_idx: int
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    num_epochs: int
    learning_rate: float
    weight_decay: float
    val_ratio: float
    grad_clip: float
    early_stop_patience: int
    label_smoothing: float
    eos_loss_weight: float
    lr_t_max: int
    seed: int
    val_gen_charts: int
    val_oracle_windows: int
    overfit_charts: int
    resume_path: Path | None
    rotations: int
    audio_augment_probability: float
    feature_noise_std: float
    max_time_mask_frames: int


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path
    level_idx: int
    start_sec: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    mert: MertConfig
    window: WindowConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


SECTIONS = {
    "路径": (PathsConfig, {
        "谱面目录": "charts_dir",
        "MERT缓存目录": "mert_cache_dir",
        "检查点目录": "checkpoint_dir",
    }),
    "MERT": (MertConfig, {
        "模型目录": "model_dir",
        "分块秒数": "chunk_sec",
        "上下文秒数": "context_sec",
        "缓存半精度": "cache_float16",
        "音频工作线程数": "audio_workers",
        "最大批大小": "batch_size",
    }),
    "滑窗": (WindowConfig, {
        "MERT帧数": "mert_frames",
        "前缀开始秒": "prefix_start_sec",
        "目标开始秒": "target_start_sec",
        "目标结束秒": "target_end_sec",
        "训练步进秒": "train_stride_sec",
        "推理步进秒": "infer_stride_sec",
    }),
    "模型": (ModelConfig, {
        "状态维度": "state",
        "注意力头数": "head",
        "层数": "layer",
        "最大词元数": "max_tokens",
    }),
    "训练": (TrainingConfig, {
        "难度编号": "level_idx",
        "批大小": "batch_size",
        "数据加载进程数": "num_workers",
        "预取批数": "prefetch_factor",
        "锁页内存": "pin_memory",
        "训练轮数": "num_epochs",
        "学习率": "learning_rate",
        "权重衰减": "weight_decay",
        "验证集比例": "val_ratio",
        "梯度裁剪": "grad_clip",
        "提前停止耐心轮数": "early_stop_patience",
        "标签平滑": "label_smoothing",
        "结束词元损失权重": "eos_loss_weight",
        "学习率退火周期": "lr_t_max",
        "随机种子": "seed",
        "整曲验证歌曲数": "val_gen_charts",
        "真值前缀验证窗口数": "val_oracle_windows",
        "过拟合歌曲数": "overfit_charts",
        "恢复检查点": "resume_path",
        "旋转增强数": "rotations",
        "音频增强概率": "audio_augment_probability",
        "特征噪声强度": "feature_noise_std",
        "最大时间遮挡帧数": "max_time_mask_frames",
    }),
    "推理": (InferenceConfig, {
        "检查点": "checkpoint",
        "难度编号": "level_idx",
        "开始秒": "start_sec",
    }),
}


def _is_type(value: Any, expected: Any) -> bool:
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is Path:
        return isinstance(value, str)
    origin = get_origin(expected)
    if origin is not None and type(None) in get_args(expected):
        return value is None or _is_type(value, next(t for t in get_args(expected) if t is not type(None)))
    return isinstance(value, expected)


def _load_section(raw: dict, section_name: str):
    cls, key_map = SECTIONS[section_name]
    section = raw.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"配置缺少对象: {section_name}")
    unknown = set(section) - set(key_map)
    missing = set(key_map) - set(section)
    if unknown or missing:
        raise ValueError(f"配置 {section_name} 键错误: 缺少={sorted(missing)} 未知={sorted(unknown)}")
    hints = get_type_hints(cls)
    values = {}
    for yaml_key, field_name in key_map.items():
        value = section[yaml_key]
        expected = hints[field_name]
        if not _is_type(value, expected):
            raise TypeError(f"配置 {section_name}.{yaml_key} 类型错误: {value!r}")
        if expected is Path or (get_origin(expected) is not None and Path in get_args(expected)):
            value = None if value is None else (ROOT_DIR / value).resolve()
        elif expected is float:
            value = float(value)
        values[field_name] = value
    return cls(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(config: AppConfig) -> None:
    mert = config.mert
    window = config.window
    model = config.model
    training = config.training
    inference = config.inference

    for value, name in (
        (window.prefix_start_sec, "滑窗.前缀开始秒"),
        (window.target_start_sec, "滑窗.目标开始秒"),
        (window.target_end_sec, "滑窗.目标结束秒"),
        (window.train_stride_sec, "滑窗.训练步进秒"),
        (window.infer_stride_sec, "滑窗.推理步进秒"),
        (training.learning_rate, "训练.学习率"),
        (training.weight_decay, "训练.权重衰减"),
        (training.val_ratio, "训练.验证集比例"),
        (training.grad_clip, "训练.梯度裁剪"),
        (training.label_smoothing, "训练.标签平滑"),
        (training.eos_loss_weight, "训练.结束词元损失权重"),
        (inference.start_sec, "推理.开始秒"),
    ):
        _require(math.isfinite(value), f"配置 {name} 必须是有限数字")

    _require(mert.model_dir.is_dir(), f"配置 MERT.模型目录不存在或不是目录: {mert.model_dir}")
    _require(math.isfinite(mert.chunk_sec) and mert.chunk_sec > 0, "配置 MERT.分块秒数 必须大于 0")
    _require(math.isfinite(mert.context_sec) and mert.context_sec >= 0, "配置 MERT.上下文秒数 不能小于 0")
    _require(mert.context_sec * 2 < mert.chunk_sec, "配置 MERT.上下文秒数的两倍必须小于分块秒数")
    _require(mert.audio_workers > 0, "配置 MERT.音频工作线程数 必须大于 0")
    _require(mert.batch_size > 0, "配置 MERT.最大批大小 必须大于 0")

    _require(window.mert_frames > 0, "配置 滑窗.MERT帧数 必须大于 0")
    _require(window.prefix_start_sec >= 0, "配置 滑窗.前缀开始秒 不能小于 0")
    _require(
        window.prefix_start_sec < window.target_start_sec < window.target_end_sec,
        "配置滑窗时间必须满足 前缀开始秒 < 目标开始秒 < 目标结束秒",
    )
    _require(window.target_end_sec <= 30.0, "配置 滑窗.目标结束秒 不能超过时间戳上限 30.0 秒")
    window_duration = window.mert_frames / 75
    _require(
        window.target_end_sec <= window_duration,
        f"配置 滑窗.目标结束秒 超出音频窗口；当前 MERT 帧数只能提供 {window_duration:.2f} 秒",
    )
    _require(window.train_stride_sec > 0, "配置 滑窗.训练步进秒 必须大于 0")
    _require(window.infer_stride_sec > 0, "配置 滑窗.推理步进秒 必须大于 0")
    target_duration = window.target_end_sec - window.target_start_sec
    _require(
        abs(window.infer_stride_sec - target_duration) <= 1e-6,
        "配置 滑窗.推理步进秒 必须等于目标区时长",
    )

    for value, name in (
        (model.state, "状态维度"),
        (model.head, "注意力头数"),
        (model.layer, "层数"),
        (model.max_tokens, "最大词元数"),
    ):
        _require(value > 0, f"配置 模型.{name} 必须大于 0")
    _require(model.state >= 4, "配置 模型.状态维度 至少为 4")
    _require(
        model.state % model.head == 0,
        "配置 模型.状态维度 必须能被 注意力头数 整除",
    )
    _require(model.max_tokens >= 2, "配置 模型.最大词元数 至少为 2，才能容纳开始和结束词元")

    _require(0 <= training.level_idx <= 6, "配置 训练.难度编号 必须在 0 到 6 之间")
    _require(training.batch_size > 0, "配置 训练.批大小 必须大于 0")
    _require(training.num_workers >= 0, "配置 训练.数据加载进程数 不能小于 0")
    _require(training.prefetch_factor > 0, "配置 训练.预取批数 必须大于 0")
    _require(training.num_epochs > 0, "配置 训练.训练轮数 必须大于 0")
    _require(training.learning_rate > 0, "配置 训练.学习率 必须大于 0")
    _require(training.weight_decay >= 0, "配置 训练.权重衰减 不能小于 0")
    _require(0 < training.val_ratio < 1, "配置 训练.验证集比例 必须大于 0 且小于 1")
    _require(training.grad_clip > 0, "配置 训练.梯度裁剪 必须大于 0")
    _require(training.early_stop_patience > 0, "配置 训练.提前停止耐心轮数 必须大于 0")
    _require(0 <= training.label_smoothing < 1, "配置 训练.标签平滑 必须大于等于 0 且小于 1")
    _require(training.eos_loss_weight > 0, "配置 训练.结束词元损失权重 必须大于 0")
    _require(training.lr_t_max > 0, "配置 训练.学习率退火周期 必须大于 0")
    _require(training.val_gen_charts >= 0, "配置 训练.整曲验证歌曲数 不能小于 0")
    _require(training.val_oracle_windows >= 0, "配置 训练.真值前缀验证窗口数 不能小于 0")
    _require(training.overfit_charts >= 0, "配置 训练.过拟合歌曲数 不能小于 0")
    _require(training.rotations in range(1, 9), "配置 训练.旋转增强数 必须在 1 到 8 之间")
    _require(0 <= training.audio_augment_probability <= 1, "配置 训练.音频增强概率 必须在 0 到 1 之间")
    _require(training.feature_noise_std >= 0, "配置 训练.特征噪声强度 不能小于 0")
    _require(training.max_time_mask_frames >= 0, "配置 训练.最大时间遮挡帧数 不能小于 0")
    _require(0 <= inference.level_idx <= 6, "配置 推理.难度编号 必须在 0 到 6 之间")
    _require(inference.start_sec >= 0, "配置 推理.开始秒 不能小于 0")


def inference_checkpoint_config(config: AppConfig) -> dict[str, dict[str, int | float]]:
    """保存会改变模型结构、输入特征或正式推理语义的配置。"""
    return {
        "MERT": {
            "模型目录": str(config.mert.model_dir),
            "分块秒数": config.mert.chunk_sec,
            "上下文秒数": config.mert.context_sec,
            "缓存半精度": config.mert.cache_float16,
        },
        "滑窗": {
            "MERT帧数": config.window.mert_frames,
            "前缀开始秒": config.window.prefix_start_sec,
            "目标开始秒": config.window.target_start_sec,
            "目标结束秒": config.window.target_end_sec,
            "推理步进秒": config.window.infer_stride_sec,
        },
        "模型": {
            "状态维度": config.model.state,
            "注意力头数": config.model.head,
            "层数": config.model.layer,
            "最大词元数": config.model.max_tokens,
        },
    }


def checkpoint_config(config: AppConfig) -> dict[str, dict[str, int | float]]:
    """保存严格续训需要保持一致的模型、任务和优化器配置。"""
    saved = inference_checkpoint_config(config)
    saved["训练任务"] = {
        "难度编号": config.training.level_idx,
        "训练步进秒": config.window.train_stride_sec,
        "验证集比例": config.training.val_ratio,
        "随机种子": config.training.seed,
        "过拟合歌曲数": config.training.overfit_charts,
        "旋转增强数": config.training.rotations,
    }
    saved["优化器"] = {
        "学习率": config.training.learning_rate,
        "权重衰减": config.training.weight_decay,
        "学习率退火周期": config.training.lr_t_max,
    }
    return saved


def validate_checkpoint_config(
    saved: dict | None,
    current: AppConfig,
    *,
    for_training: bool,
) -> None:
    expected = checkpoint_config(current) if for_training else inference_checkpoint_config(current)
    complete = saved is not None and all(
        section in saved and all(key in saved[section] for key in values)
        for section, values in expected.items()
    )
    if not complete:
        raise ValueError("检查点配置不完整或来自旧架构，拒绝加载")
    for section, values in expected.items():
        saved_section = saved.get(section, {})
        for key, value in values.items():
            if saved_section.get(key) != value:
                raise ValueError(
                    f"检查点与当前配置不兼容: {section}.{key}="
                    f"{saved_section.get(key)!r}，当前为 {value!r}"
                )


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"无法读取配置文件 {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"配置文件 YAML 格式错误: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("配置文件根节点必须是对象")
    unknown = set(raw) - set(SECTIONS)
    missing = set(SECTIONS) - set(raw)
    if unknown or missing:
        raise ValueError(f"配置分组错误: 缺少={sorted(missing)} 未知={sorted(unknown)}")
    loaded = {name: _load_section(raw, name) for name in SECTIONS}
    config = AppConfig(*loaded.values())
    _validate_config(config)
    return config


CONFIG = load_config()


def _expect_error(message: str, action) -> None:
    try:
        action()
    except Exception as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"预期异常: {message}")


def _self_check() -> None:
    from infer import _level_arg, _nonnegative_float_arg, _prefix_relative_cs, frames_to_prefix_tokens
    from maidata_parser import Frame, Note, NoteType, TapType
    from tokenizer import TS_BASE

    assert CONFIG.paths.charts_dir == ROOT_DIR / "charts"
    assert CONFIG.mert.model_dir == ROOT_DIR / "MERT-v1-95M"
    assert CONFIG.model.state == 768
    assert CONFIG.window.mert_frames == 2612
    assert CONFIG.window.target_end_sec - CONFIG.window.target_start_sec == 10.0
    assert CONFIG.model.max_tokens == 2048
    assert CONFIG.training.level_idx == 5
    assert load_config() == CONFIG

    window_start = 10.0
    just_below_prefix = Frame((Note(NoteType.TAP, TapType.LANE1),), 15.9949)
    rounded_to_prefix = Frame((Note(NoteType.TAP, TapType.LANE2),), 15.999999999999998)
    just_below_target = Frame((Note(NoteType.TAP, TapType.LANE3),), 21.9949)
    rounded_to_target = Frame((Note(NoteType.TAP, TapType.LANE4),), 21.999999999999998)
    assert _prefix_relative_cs(just_below_prefix, window_start) is None
    assert _prefix_relative_cs(rounded_to_prefix, window_start) == 600
    assert _prefix_relative_cs(just_below_target, window_start) == 1199
    assert _prefix_relative_cs(rounded_to_target, window_start) is None
    prefix_tokens = frames_to_prefix_tokens(
        [just_below_prefix, rounded_to_prefix, just_below_target, rounded_to_target],
        window_start,
    )
    assert prefix_tokens.count(TS_BASE + 600) == 1
    assert prefix_tokens.count(TS_BASE + 1199) == 1

    _expect_error(
        "目标结束秒 超出音频窗口",
        lambda: _validate_config(replace(CONFIG, window=replace(CONFIG.window, mert_frames=1000))),
    )
    _expect_error(
        "状态维度 必须能被",
        lambda: _validate_config(replace(CONFIG, model=replace(CONFIG.model, head=5))),
    )
    _expect_error(
        "状态维度 至少为 4",
        lambda: _validate_config(replace(CONFIG, model=replace(CONFIG.model, state=2, head=1))),
    )
    _expect_error(
        "音频增强概率",
        lambda: _validate_config(replace(
            CONFIG,
            training=replace(CONFIG.training, audio_augment_probability=1.1),
        )),
    )
    _expect_error(
        "最大时间遮挡帧数",
        lambda: _validate_config(replace(
            CONFIG,
            training=replace(CONFIG.training, max_time_mask_frames=-1),
        )),
    )
    _expect_error(
        "时间戳上限",
        lambda: _validate_config(replace(
            CONFIG,
            window=replace(CONFIG.window, target_end_sec=30.01, infer_stride_sec=18.01),
        )),
    )

    saved = checkpoint_config(CONFIG)
    _expect_error(
        "滑窗.MERT帧数",
        lambda: validate_checkpoint_config(
            saved, replace(CONFIG, window=replace(CONFIG.window, mert_frames=2000)),
            for_training=False,
        ),
    )
    validate_checkpoint_config(
        saved, replace(CONFIG, window=replace(CONFIG.window, train_stride_sec=2.0)),
        for_training=False,
    )
    _expect_error(
        "优化器.学习率",
        lambda: validate_checkpoint_config(
            saved, replace(CONFIG, training=replace(CONFIG.training, learning_rate=0.001)),
            for_training=True,
        ),
    )
    _expect_error(
        "训练任务.难度编号",
        lambda: validate_checkpoint_config(
            saved, replace(CONFIG, training=replace(CONFIG.training, level_idx=4)),
            for_training=True,
        ),
    )
    _expect_error(
        "旧架构",
        lambda: validate_checkpoint_config(None, CONFIG, for_training=False),
    )
    partial = checkpoint_config(CONFIG)
    del partial["优化器"]
    _expect_error(
        "旧架构",
        lambda: validate_checkpoint_config(partial, CONFIG, for_training=True),
    )

    for action in (
        lambda: _level_arg("-1"),
        lambda: _level_arg("7"),
        lambda: _nonnegative_float_arg("nan"),
        lambda: _nonnegative_float_arg("-1"),
    ):
        _expect_error("", action)
    print("[config] 自检通过")


if __name__ == "__main__":
    _self_check()
