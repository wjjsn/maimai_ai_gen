"""持久化滑窗谱面索引；数据或窗口配置变化时自动重建。"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from config import CONFIG
from maidata_parser import _match_music, compiler, load_music_data, music_data_version
from mel_cache import main as rebuild_mel_cache
from tokenizer import EOS, SOS, encode_frame

CACHE_VERSION = 4
PREFIX_START_SEC = CONFIG.window.prefix_start_sec
TARGET_START_SEC = CONFIG.window.target_start_sec
TARGET_END_SEC = CONFIG.window.target_end_sec
ENTRY_DTYPE = np.dtype([
    ("chart_id", "<u4"),
    ("source_start_frame", "<i8"),
    ("source_end_frame", "<i8"),
    ("left_pad_frames", "<i8"),
    ("window_start_sec", "<f8"),
    ("target_start_sec", "<f8"),
    ("token_start", "<i8"),
    ("token_length", "<u4"),
    ("loss_start", "<u4"),
])


def _level_arg(value: str) -> int:
    level = int(value)
    if not 0 <= level <= 6:
        raise argparse.ArgumentTypeError("难度编号必须在 0 到 6 之间")
    return level


def _positive_float_arg(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return number


def _state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _config(level_idx: int, sample_rate: int, hop_length: int, n_mels: int, stride_sec: float, mel_frames: int, music_version: str) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "level_idx": level_idx,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "stride_sec": stride_sec,
        "mel_frames": mel_frames,
        "prefix_start_sec": PREFIX_START_SEC,
        "target_start_sec": TARGET_START_SEC,
        "target_end_sec": TARGET_END_SEC,
        "music_data_version": music_version,
    }


def _key(config: dict) -> str:
    data = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _paths(cache_dir: Path, config: dict) -> tuple[Path, Path]:
    root = cache_dir.parent / "chart_index"
    cache_root = root / _key(config)
    return cache_root, root / f"{_key(config)}.lock"


def _current_path(cache_root: Path) -> Path | None:
    pointer = cache_root / "current.json"
    try:
        generation = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
        path = cache_root / generation
        return path if path.is_dir() else None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _publish(cache_root: Path, build_path: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    final_path = cache_root / build_path.name
    build_path.replace(final_path)
    pointer = cache_root / "current.json"
    temp = pointer.with_name(f".{pointer.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps({"generation": final_path.name}), encoding="utf-8")
    temp.replace(pointer)


def _scan_sources(charts_dir: Path, mel_dir: Path) -> list[dict]:
    from mel_cache import cache_key

    sources = []
    for chart_path in sorted(charts_dir.rglob("maidata.txt")):
        chart_dir = chart_path.parent
        track_path = chart_dir / "track.mp3"
        rel = chart_dir.relative_to(charts_dir)
        mel_path = mel_dir / f"{cache_key(str(rel))}.npy"
        if track_path.exists() and mel_path.exists():
            sources.append({
                "chart": str(chart_path.relative_to(charts_dir)),
                "mel": str(mel_path.relative_to(mel_dir.parent)),
                "chart_state": _state(chart_path),
                "mel_state": _state(mel_path),
            })
    return sources


def _is_current(cache_path: Path, config: dict, sources: list[dict]) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not (cache_path / "COMPLETE").exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["config"] == config and manifest["sources"] == sources
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as file:
                json.dump({"pid": os.getpid(), "created_ns": time.time_ns()}, file)
            return
        except FileExistsError:
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(lock["pid"])
                stale = time.time_ns() - int(lock["created_ns"]) > 3_600_000_000_000
                if stale or not Path(f"/proc/{pid}").exists():
                    lock_path.unlink(missing_ok=True)
                    continue
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                # ponytail: 损坏锁无法判断所有者，直接回收；跨主机共享缓存时改为文件锁。
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.2)


def _compile(charts_dir: Path, cache_dir: Path, build_path: Path, config: dict, sources: list[dict]) -> None:
    frames_per_sec = config["sample_rate"] / config["hop_length"]
    entries = []
    tokens: list[int] = []
    charts = []
    songs = load_music_data()
    excluded = 0
    for source in sources:
        chart_path = charts_dir / source["chart"]
        mel_path = cache_dir.parent / source["mel"]
        try:
            text = chart_path.read_text(encoding="utf-8")
            parser = compiler(hop_length=config["hop_length"], sample_rate=config["sample_rate"])
            parser.parse(text, music_data=songs)
        except Exception as error:
            raise RuntimeError(f"谱面解析失败，终止缓存构建: {source['chart']}") from error
        song = _match_music(text, parser.chart.title, songs)
        if song is not None and song.get("basic_info", {}).get("genre") == "宴会場":
            excluded += 1
            continue
        level = parser.chart.all_levels[config["level_idx"]]
        if level is None:
            continue
        chart_id = len(charts)
        charts.append({"chart": source["chart"], "mel": source["mel"]})
        mel_total_frames = np.load(mel_path, mmap_mode="r").shape[1]
        target_start = 0.0
        while target_start < mel_total_frames / frames_per_sec:
            window_start = target_start - TARGET_START_SEC
            logical_start = round(window_start * frames_per_sec)
            source_start = max(0, logical_start)
            source_end = min(mel_total_frames, logical_start + config["mel_frames"])
            row = [SOS]
            loss_start = 1
            loss_started = False
            for frame in level.frames:
                if not frame.notes:
                    continue
                rel_cs = round((frame.time_sec - window_start) * 100)
                prefix = round(PREFIX_START_SEC * 100) <= rel_cs < round(TARGET_START_SEC * 100)
                target = round(TARGET_START_SEC * 100) <= rel_cs < round(TARGET_END_SEC * 100)
                if not (prefix or target):
                    continue
                if target and not loss_started:
                    loss_start = len(row)
                    loss_started = True
                row.extend(encode_frame(frame, rel_cs / 100.0))
            if not loss_started:
                loss_start = len(row)
            row.append(EOS)
            token_start = len(tokens)
            tokens.extend(row)
            entries.append((chart_id, source_start, max(source_start, source_end), source_start - logical_start,
                            window_start, target_start, token_start, len(row), loss_start))
            target_start += config["stride_sec"]
    build_path.mkdir(parents=True)
    np.save(build_path / "entries.npy", np.array(entries, dtype=ENTRY_DTYPE))
    np.save(build_path / "tokens.npy", np.asarray(tokens, dtype=np.uint16))
    manifest = {"config": config, "sources": sources, "charts": charts, "windows": len(entries)}
    (build_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    (build_path / "COMPLETE").touch()
    print(f"[chart-cache] 宴会場排除 {excluded} 首")


def ensure_chart_cache(charts_dir: str | Path, cache_dir: str | Path, *, level_idx: int, sample_rate: int, n_fft: int, hop_length: int, n_mels: int, stride_sec: float, mel_frames: int, build_mel: bool = True) -> Path:
    charts_dir = Path(charts_dir)
    cache_dir = Path(cache_dir)
    if build_mel:
        rebuild_mel_cache(charts_dir, cache_dir, sample_rate, n_fft, hop_length, n_mels)
    config = _config(level_idx, sample_rate, hop_length, n_mels, stride_sec, mel_frames, music_data_version())
    cache_root, lock_path = _paths(cache_dir, config)
    sources = _scan_sources(charts_dir, cache_dir)
    current_path = _current_path(cache_root)
    if current_path is not None and _is_current(current_path, config, sources):
        return current_path
    _acquire_lock(lock_path)
    try:
        sources = _scan_sources(charts_dir, cache_dir)
        current_path = _current_path(cache_root)
        if current_path is not None and _is_current(current_path, config, sources):
            return current_path
        temp_path = cache_root / f".build-{os.getpid()}-{time.time_ns()}"
        shutil.rmtree(temp_path, ignore_errors=True)
        print(f"[chart-cache] 索引失效，正在重建 {_key(config)}...")
        _compile(charts_dir, cache_dir, temp_path, config, sources)
        _publish(cache_root, temp_path)
        return _current_path(cache_root)
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", type=Path, default=CONFIG.paths.charts_dir)
    parser.add_argument("--cache-dir", type=Path, default=CONFIG.paths.mel_cache_dir)
    parser.add_argument("--level", type=_level_arg, default=CONFIG.training.level_idx)
    parser.add_argument("--stride-sec", type=_positive_float_arg, default=CONFIG.window.train_stride_sec)
    args = parser.parse_args()
    path = ensure_chart_cache(
        args.charts_dir,
        args.cache_dir,
        level_idx=args.level,
        sample_rate=CONFIG.audio.sample_rate,
        n_fft=CONFIG.audio.n_fft,
        hop_length=CONFIG.audio.hop_length,
        n_mels=CONFIG.audio.n_mels,
        stride_sec=args.stride_sec,
        mel_frames=CONFIG.window.mel_frames,
    )
    print(f"[chart-cache] 就绪: {path}")


if __name__ == "__main__":
    main()
