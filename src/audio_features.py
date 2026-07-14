"""现场解码音频，冻结 MERT 并按 200Hz 谱面时间轴输出特征。"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from config import CONFIG


MERT_ID = "m-a-p/MERT-v1-95M"
MERT_HIDDEN_DIM = 768
_MERT: tuple[AutoModel, Wav2Vec2FeatureExtractor, int, int] | None = None


@dataclass(frozen=True)
class AudioAugmentation:
    pitch_steps: float = 0.0
    speed: float = 1.0
    shift_sec: float = 0.0
    frequency_mask_bins: int = 0
    eq_gain: float = 0.0
    noise_std: float = 0.0


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(()).item()


def sample_augmentation() -> AudioAugmentation:
    config = CONFIG.augmentation
    return AudioAugmentation(
        _uniform(-config.max_pitch_steps, config.max_pitch_steps) if torch.rand(()) < config.pitch_probability else 0.0,
        _uniform(config.min_speed, config.max_speed) if torch.rand(()) < config.speed_probability else 1.0,
        _uniform(-config.max_shift_sec, config.max_shift_sec) if torch.rand(()) < config.shift_probability else 0.0,
        0, 0.0,
        _uniform(0.0, config.max_noise_std) if torch.rand(()) < config.noise_probability else 0.0,
    )


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        return torchaudio.load(str(path))
    except RuntimeError as original:
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "24000", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        samples = np.frombuffer(result.stdout, dtype="<f4").copy()
        if samples.size == 0:
            raise RuntimeError(f"音频解码失败: {path}: {result.stderr.decode(errors='replace').strip()}") from original
        return torch.from_numpy(samples).unsqueeze(0), 24000


def _time_stretch(waveform: torch.Tensor, speed: float) -> torch.Tensor:
    if abs(speed - 1.0) < 1e-6:
        return waveform
    n_fft, hop_length = 512, 128
    window = torch.hann_window(n_fft, device=waveform.device)
    spectrum = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    phase = torch.linspace(0, torch.pi * hop_length, spectrum.shape[-2], device=waveform.device)[..., None]
    return torch.istft(torchaudio.functional.phase_vocoder(spectrum, speed, phase), n_fft=n_fft, hop_length=hop_length, window=window, length=max(1, round(waveform.shape[-1] / speed)))


def _augment_waveform(waveform: torch.Tensor, augmentation: AudioAugmentation, sample_rate: int) -> torch.Tensor:
    if augmentation.pitch_steps:
        waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, augmentation.pitch_steps, n_fft=512, hop_length=128)
    waveform = _time_stretch(waveform, augmentation.speed)
    if augmentation.noise_std:
        waveform = waveform + torch.randn_like(waveform) * augmentation.noise_std
    shift = round(augmentation.shift_sec * sample_rate)
    if shift > 0:
        return torch.nn.functional.pad(waveform, (shift, 0))[..., :waveform.shape[-1]]
    if shift < 0:
        return torch.nn.functional.pad(waveform[..., -shift:], (0, -shift))
    return waveform


def _mert(device: torch.device) -> tuple[AutoModel, Wav2Vec2FeatureExtractor, int, int]:
    global _MERT
    if _MERT is None:
        model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
        step, receptive = 1, 1
        for stride, kernel in zip(model.config.conv_stride, model.config.conv_kernel):
            receptive += (kernel - 1) * step
            step *= stride
        if model.config.hidden_size != MERT_HIDDEN_DIM:
            raise ValueError(f"MERT 隐藏维度错误: {model.config.hidden_size}")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _MERT = model, processor, step, receptive
    model, processor, step, receptive = _MERT
    return model.to(device), processor, step, receptive


def _interpolate_to_chart_axis(features: np.ndarray, centers: np.ndarray, samples: int, sample_rate: int) -> np.ndarray:
    length = round(samples / sample_rate * CONFIG.audio.frames_per_sec) + 1
    target = np.arange(length, dtype=np.float64) / CONFIG.audio.frames_per_sec
    right = np.searchsorted(centers, target).clip(1, len(centers) - 1)
    left = right - 1
    alpha = ((target - centers[left]) / (centers[right] - centers[left]))[:, None]
    result = features[left] * (1 - alpha) + features[right] * alpha
    result[target <= centers[0]] = features[0]
    result[target >= centers[-1]] = features[-1]
    return result.astype(np.float32, copy=False)


def _extract_mert(waveform: torch.Tensor, sample_rate: int, device: torch.device) -> np.ndarray:
    model, processor, step, receptive = _mert(device)
    if sample_rate != processor.sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, processor.sampling_rate)
        sample_rate = processor.sampling_rate
    values = processor(waveform.squeeze(0).cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values[0]
    output, centers = [], []
    core_samples, context_samples = sample_rate * 20, sample_rate * 2
    for core_start in range(0, len(values), core_samples):
        core_end = min(len(values), core_start + core_samples)
        start, end = max(0, core_start - context_samples), min(len(values), core_end + context_samples)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(values[start:end].unsqueeze(0).to(device), output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy()
        token_centers = (start + (receptive - 1) / 2 + np.arange(len(hidden)) * step) / sample_rate
        keep = (token_centers >= core_start / sample_rate) & (token_centers < core_end / sample_rate)
        output.append(hidden[keep])
        centers.append(token_centers[keep])
    return _interpolate_to_chart_axis(np.concatenate(output), np.concatenate(centers), len(values), sample_rate)


def extract_audio_features(path: str | Path, device: torch.device | None = None, augmentation: AudioAugmentation | None = None) -> np.ndarray:
    waveform, source_rate = load_audio(path)
    waveform = waveform.mean(dim=0, keepdim=True).float()
    if augmentation is not None:
        waveform = _augment_waveform(waveform, augmentation, source_rate)
    return _extract_mert(waveform, source_rate, device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def _self_check() -> None:
    features = np.array([[0.0], [10.0]], dtype=np.float32)
    aligned = _interpolate_to_chart_axis(features, np.array([0.01, 0.03]), 2400, 24000)
    assert aligned.shape == (21, 1) and np.allclose(aligned[[0, 4, -1], 0], (0, 5, 10))
    print("[audio-features] 自检通过")
