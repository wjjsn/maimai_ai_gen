from dataclasses import dataclass, fields
import math
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    charts_dir: Path
    checkpoint_dir: Path
    train_output_dir: Path
    inference_output_dir: Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    hop_length: int
    n_fft: int
    win_length: int
    n_mels: int
    f_min: float
    f_max: float

    @property
    def frames_per_sec(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass(frozen=True)
class AugmentationConfig:
    pitch_probability: float
    max_pitch_steps: float
    speed_probability: float
    min_speed: float
    max_speed: float
    shift_probability: float
    max_shift_sec: float
    frequency_mask_probability: float
    max_frequency_mask_bins: int
    eq_probability: float
    max_eq_gain: float
    noise_probability: float
    max_noise_std: float


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    layers: int
    attention_heads: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    window_frames: int
    stride: int
    song_limit: int
    batch_size: int
    pin_memory: bool
    num_workers: int
    prefetch_factor: int
    resume_checkpoint: Path | None
    num_epochs: int
    learning_rate: float
    min_learning_rate: float
    warmup_ratio: float
    weight_decay: float
    val_ratio: float
    test_ratio: float
    grad_clip: float
    early_stop_patience: int
    seed: int
    event_window_ratio: float


@dataclass(frozen=True)
class InferenceConfig:
    audio_path: Path
    checkpoint: Path
    level_idx: int
    level_query: float
    short_min_gap_frames: int
    long_min_frames: int
    min_duration_sec: float
    max_duration_sec: float


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    audio: AudioConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


SECTIONS = {
    "路径": (PathsConfig, {
        "谱面目录": "charts_dir", "检查点目录": "checkpoint_dir",
        "训练输出目录": "train_output_dir", "推理输出目录": "inference_output_dir",
    }),
    "音频": (AudioConfig, {
        "采样率": "sample_rate", "跳步长度": "hop_length", "FFT长度": "n_fft",
        "窗口长度": "win_length", "Mel频带数": "n_mels", "最低频率": "f_min",
        "最高频率": "f_max",
    }),
    "数据增强": (AugmentationConfig, {
        "变调概率": "pitch_probability", "最大变调半音": "max_pitch_steps",
        "变速概率": "speed_probability", "最小变速倍率": "min_speed",
        "最大变速倍率": "max_speed", "时间平移概率": "shift_probability",
        "最大平移秒数": "max_shift_sec", "频率遮蔽概率": "frequency_mask_probability",
        "最大遮蔽频带数": "max_frequency_mask_bins", "EQ概率": "eq_probability",
        "最大EQ增益": "max_eq_gain", "噪声概率": "noise_probability",
        "最大噪声标准差": "max_noise_std",
    }),
    "模型": (ModelConfig, {
        "隐藏维度": "hidden_dim", "层数": "layers",
        "注意力头数": "attention_heads", "丢弃率": "dropout",
    }),
    "训练": (TrainingConfig, {
        "窗口帧数": "window_frames", "窗口步长": "stride", "训练歌曲数": "song_limit",
        "批大小": "batch_size",
        "锁页内存": "pin_memory", "数据加载进程数": "num_workers",
        "每进程预取批次数": "prefetch_factor", "恢复检查点": "resume_checkpoint",
        "训练轮数": "num_epochs",
        "学习率": "learning_rate", "最低学习率": "min_learning_rate", "权重衰减": "weight_decay",
        "预热比例": "warmup_ratio",
        "验证集比例": "val_ratio", "测试集比例": "test_ratio", "梯度裁剪": "grad_clip",
        "提前停止耐心轮数": "early_stop_patience", "随机种子": "seed",
        "事件窗口比例": "event_window_ratio",
    }),
    "推理": (InferenceConfig, {
        "输入音频": "audio_path", "检查点": "checkpoint", "难度编号": "level_idx",
        "浮点难度": "level_query",
        "短音最小间隔帧": "short_min_gap_frames", "持续音最短帧数": "long_min_frames",
        "最短持续秒数": "min_duration_sec", "最长持续秒数": "max_duration_sec",
    }),
}


def _matches(value: Any, expected: Any) -> bool:
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is Path:
        return isinstance(value, str)
    origin = get_origin(expected)
    if origin in (UnionType,):
        return any(_matches(value, item) for item in get_args(expected))
    return isinstance(value, expected)


def _load_section(raw: dict, name: str):
    cls, mapping = SECTIONS[name]
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"配置缺少对象: {name}")
    unknown = set(section) - set(mapping)
    missing = set(mapping) - set(section)
    if unknown or missing:
        raise ValueError(f"配置 {name} 键错误: 缺少={sorted(missing)} 未知={sorted(unknown)}")
    hints = get_type_hints(cls)
    values = {}
    for yaml_name, field_name in mapping.items():
        value = section[yaml_name]
        expected = hints[field_name]
        if not _matches(value, expected):
            raise TypeError(f"配置 {name}.{yaml_name} 类型错误: {value!r}")
        if expected is Path or Path in get_args(expected):
            if value is None:
                values[field_name] = None
                continue
            value = (ROOT_DIR / value).resolve()
        elif expected is float:
            value = float(value)
        values[field_name] = value
    return cls(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate(config: AppConfig) -> None:
    audio, augmentation = config.audio, config.augmentation
    model, training, inference = config.model, config.training, config.inference
    for group in (audio, augmentation, model, training, inference):
        for item in fields(group):
            value = getattr(group, item.name)
            if isinstance(value, float):
                _require(math.isfinite(value), f"配置 {item.name} 必须是有限数字")
    _require(audio.sample_rate > 0 and audio.hop_length > 0, "音频采样率和跳步长度必须大于 0")
    _require(audio.n_fft >= audio.win_length > 0, "音频 FFT长度必须不小于窗口长度")
    _require(audio.n_mels > 0 and 0 <= audio.f_min < audio.f_max <= audio.sample_rate / 2, "音频频率范围无效")
    probabilities = (
        augmentation.pitch_probability, augmentation.speed_probability,
        augmentation.shift_probability, augmentation.frequency_mask_probability,
        augmentation.eq_probability, augmentation.noise_probability,
    )
    _require(all(0 <= value <= 1 for value in probabilities), "数据增强概率必须在 [0, 1] 内")
    _require(augmentation.max_pitch_steps >= 0, "最大变调半音不能为负")
    _require(0 < augmentation.min_speed <= augmentation.max_speed, "变速倍率范围无效")
    _require(augmentation.max_shift_sec >= 0, "最大平移秒数不能为负")
    _require(0 <= augmentation.max_frequency_mask_bins <= audio.n_mels, "频率遮蔽范围无效")
    _require(augmentation.max_eq_gain >= 0 and augmentation.max_noise_std >= 0, "EQ 或噪声强度不能为负")
    _require(model.hidden_dim > 0 and model.layers > 0 and model.attention_heads > 0, "模型维度、层数和注意力头数必须大于 0")
    _require(model.hidden_dim % model.attention_heads == 0, "隐藏维度必须能被注意力头数整除")
    _require(0 <= model.dropout < 1, "模型丢弃率必须在 [0, 1) 内")
    _require(0 <= inference.level_idx <= 6, "难度编号必须在 0 到 6 之间")
    _require(0 < inference.level_query <= 15, "推理浮点难度必须在 (0, 15] 内")
    _require(training.window_frames > 0 and training.stride > 0, "训练窗口和步长必须大于 0")
    _require(training.song_limit >= 0, "训练歌曲数不能为负")
    _require(training.batch_size > 0, "批大小必须大于 0")
    _require(training.num_workers >= 0, "数据加载进程数不能为负")
    _require(training.prefetch_factor > 0, "每进程预取批次数必须大于 0")
    _require(training.num_epochs > 0 and training.learning_rate > 0 and training.min_learning_rate >= 0, "学习率或训练轮数无效")
    _require(training.min_learning_rate <= training.learning_rate, "最低学习率不能高于学习率")
    _require(0 <= training.warmup_ratio < 1, "预热比例必须在 [0, 1) 内")
    _require(0 < training.val_ratio < 1 and 0 < training.test_ratio < 1 and training.val_ratio + training.test_ratio < 1, "数据集比例无效")
    _require(training.grad_clip > 0 and training.early_stop_patience > 0, "训练控制参数无效")
    _require(0 <= training.event_window_ratio <= 1, "事件窗口比例必须在 [0, 1] 内")
    _require(inference.short_min_gap_frames >= 0 and inference.long_min_frames > 0, "推理帧数阈值无效")
    _require(0 < inference.min_duration_sec <= inference.max_duration_sec, "持续音时长范围无效")


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
    config = AppConfig(
        loaded["路径"], loaded["音频"], loaded["数据增强"],
        loaded["模型"], loaded["训练"], loaded["推理"],
    )
    _validate(config)
    return config


CONFIG = load_config()


def checkpoint_config(config: AppConfig = CONFIG) -> dict:
    return {
        "audio": vars(config.audio),
        "model": vars(config.model),
        "window_frames": config.training.window_frames,
    }


def _self_check() -> None:
    assert CONFIG.audio.frames_per_sec == 200
    assert CONFIG.paths.charts_dir == ROOT_DIR / "charts"
    assert load_config() == CONFIG
    assert set(vars(CONFIG.model)) == {"hidden_dim", "layers", "attention_heads", "dropout"}
    assert CONFIG.training.stride == 512 and CONFIG.training.song_limit == 100
    assert CONFIG.training.batch_size == 48
    assert CONFIG.training.num_workers == 4 and CONFIG.training.prefetch_factor == 2
    assert CONFIG.training.resume_checkpoint is None
    assert CONFIG.inference.level_query == 13.0
    print("[config] 自检通过")


if __name__ == "__main__":
    _self_check()
