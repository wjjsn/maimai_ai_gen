"""持久化滑窗谱面索引；优先复用未变化歌曲，仅增量编译。"""

import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CONFIG
from constrained_decode import validate_frames
from maidata_parser import _match_music, compiler, load_music_data, music_data_version
from mert_cache import main as rebuild_mert_cache
from tokenizer import EOS, SOS, encode_frame

CACHE_VERSION = 14
MERT_FRAMES_PER_SEC = 75
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
_PREFLIGHT_SONGS: list[dict] = []
_PREFLIGHT_CONFIG: dict = {}


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


def _config(level_idx: int, stride_sec: float, mert_frames: int, music_version: str) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "level_idx": level_idx,
        "stride_sec": stride_sec,
        "mert_frames": mert_frames,
        "prefix_start_sec": PREFIX_START_SEC,
        "target_start_sec": TARGET_START_SEC,
        "target_end_sec": TARGET_END_SEC,
        "max_tokens": CONFIG.model.max_tokens,
        "validation_timeout_sec": CONFIG.mert.validation_timeout_sec,
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


def _scan_sources(charts_dir: Path, mel_dir: Path, allowed_charts: set[str] | None = None) -> list[dict]:
    from mert_cache import cache_key

    sources = []
    for chart_path in sorted(charts_dir.rglob("maidata.txt")):
        chart_dir = chart_path.parent
        track_path = chart_dir / "track.mp3"
        rel = chart_dir.relative_to(charts_dir)
        chart = str(chart_path.relative_to(charts_dir))
        if allowed_charts is not None and chart not in allowed_charts:
            continue
        mel_path = mel_dir / f"{cache_key(str(rel))}.npy"
        if track_path.exists() and mel_path.exists():
            sources.append({
                "chart": chart,
                "mel": str(mel_path.relative_to(mel_dir.parent)),
                "chart_state": _state(chart_path),
                "mel_state": _state(mel_path),
            })
    return sources


def _scan_chart_sources(charts_dir: Path, mel_dir: Path) -> list[dict]:
    """列出全部可配对歌曲；MERT 尚未生成的歌曲也交给 CPU 提前解析。"""
    from mert_cache import cache_key

    sources = []
    for chart_path in sorted(charts_dir.rglob("maidata.txt")):
        chart_dir = chart_path.parent
        if not (chart_dir / "track.mp3").exists():
            continue
        rel = chart_dir.relative_to(charts_dir)
        mel_path = mel_dir / f"{cache_key(str(rel))}.npy"
        sources.append({
            "chart": str(chart_path.relative_to(charts_dir)),
            "mel": str(mel_path.relative_to(mel_dir.parent)),
            "chart_state": _state(chart_path),
        })
    return sources


def _is_current(cache_path: Path, config: dict, input_sources: list[dict]) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not (cache_path / "COMPLETE").exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["config"] == config and manifest.get("input_sources") == input_sources
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _load_reusable(cache_path: Path | None, config: dict, sources: list[dict]) -> dict[str, tuple[dict, np.ndarray, np.ndarray]]:
    """返回谱面和 MERT 均未变化的旧窗口；支持从 v8 generation 无损迁移。"""
    if cache_path is None:
        return {}
    try:
        manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
        old_config = dict(manifest["config"])
        old_config["cache_version"] = config["cache_version"]
        old_config.setdefault("max_tokens", config["max_tokens"])
        old_config.setdefault("validation_timeout_sec", config["validation_timeout_sec"])
        if old_config != config:
            return {}
        old_sources = {item["chart"]: item for item in manifest["sources"]}
        old_charts = {item["chart"]: (index, item) for index, item in enumerate(manifest["charts"])}
        entries = np.load(cache_path / "entries.npy", mmap_mode="r")
        tokens = np.load(cache_path / "tokens.npy", mmap_mode="r")
        reusable = {}
        for source in sources:
            old_source = old_sources.get(source["chart"])
            old_chart = old_charts.get(source["chart"])
            if old_source == source and old_chart is not None:
                reusable[source["chart"]] = (old_chart[1], entries[entries["chart_id"] == old_chart[0]], tokens)
        return reusable
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _find_prior_cache(cache_dir: Path, config: dict) -> Path | None:
    """找一个仅缓存版本不同的已完成 generation，供增量迁移复用。"""
    root = cache_dir.parent / "chart_index"
    for pointer in root.glob("*/current.json"):
        try:
            generation = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
            path = pointer.parent / generation
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            old_config = dict(manifest["config"])
            old_config["cache_version"] = config["cache_version"]
            old_config.setdefault("max_tokens", config["max_tokens"])
            old_config.setdefault("validation_timeout_sec", config["validation_timeout_sec"])
            if (path / "COMPLETE").exists() and old_config == config:
                return path
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


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


def _init_preflight(songs: list[dict], config: dict) -> None:
    global _PREFLIGHT_SONGS, _PREFLIGHT_CONFIG
    _PREFLIGHT_SONGS = songs
    _PREFLIGHT_CONFIG = config


def _parse_source(charts_dir: Path, source: dict):
    chart_path = charts_dir / source["chart"]
    try:
        text = chart_path.read_text(encoding="utf-8")
        parser = compiler()
        parser.parse(text, music_data=_PREFLIGHT_SONGS)
        song = _match_music(text, parser.chart.title, _PREFLIGHT_SONGS)
        result = (parser.chart.all_levels, song, parser._warned)
        eligible, reason = _eligible_source(source, result, None, _PREFLIGHT_CONFIG)
        return (result if eligible else None), reason
    except Exception as error:
        return None, str(error)


def _max_window_tokens(level, stride_sec: float) -> int:
    """不依赖 MERT 长度预检所有含谱面窗口，避免无效歌曲先占用特征缓存。"""
    frames = [frame for frame in level.frames if frame.notes]
    if not frames:
        return 2
    max_time = max(frame.time_sec for frame in frames)
    target_start = 0.0
    max_length = 2
    # 谱面时间不会超过音频；只预检实际可能提交的目标起点。
    while target_start <= max_time:
        window_start = target_start - TARGET_START_SEC
        length = 2  # SOS + EOS
        for frame in frames:
            rel_cs = round((frame.time_sec - window_start) * 100)
            if round(PREFIX_START_SEC * 100) <= rel_cs < round(TARGET_END_SEC * 100):
                length += len(encode_frame(frame, rel_cs / 100.0))
        max_length = max(max_length, length)
        target_start += stride_sec
    return max_length


def _eligible_source(
    source: dict,
    result: tuple[list, dict | None, set[str]] | None,
    error: str | None,
    config: dict,
) -> tuple[bool, str | None]:
    if result is None:
        return False, error
    levels, song, _warnings = result
    if song is not None and song.get("basic_info", {}).get("genre") == "宴会場":
        return False, "宴会場"
    level = levels[config["level_idx"]]
    if level is None:
        return False, "缺少目标难度"
    if validate_frames(level.frames, CONFIG.mert.validation_timeout_sec):
        return False, "严格规则违规"
    max_length = _max_window_tokens(level, config["stride_sec"])
    if max_length > config["max_tokens"]:
        return False, f"窗口 token 数 {max_length} 超过上限 {config['max_tokens']}"
    return True, None


def _append_reused_chart(
    chart: dict,
    old_rows: np.ndarray,
    old_tokens: np.ndarray,
    chart_id: int,
    entries: list,
    tokens: list[int],
) -> None:
    for old_row in old_rows:
        token_start = int(old_row["token_start"])
        token_length = int(old_row["token_length"])
        entries.append((
            chart_id,
            int(old_row["source_start_frame"]),
            int(old_row["source_end_frame"]),
            int(old_row["left_pad_frames"]),
            float(old_row["window_start_sec"]),
            float(old_row["target_start_sec"]),
            len(tokens), token_length, int(old_row["loss_start"]),
        ))
        tokens.extend(old_tokens[token_start:token_start + token_length])


def _compile_source(cache_dir: Path, source: dict, result: tuple[list, dict | None, set[str]]):
    """在预检子进程中生成一首歌的局部窗口，主进程只负责拼接偏移。"""
    levels, _song, _warnings = result
    level = levels[_PREFLIGHT_CONFIG["level_idx"]]
    mel_path = cache_dir.parent / source["mel"]
    mel_total_frames = np.load(mel_path, mmap_mode="r").shape[0]
    entries = []
    tokens: list[int] = []
    target_start = 0.0
    while target_start < mel_total_frames / MERT_FRAMES_PER_SEC:
        window_start = target_start - TARGET_START_SEC
        logical_start = round(window_start * MERT_FRAMES_PER_SEC)
        source_start = max(0, logical_start)
        source_end = min(mel_total_frames, logical_start + _PREFLIGHT_CONFIG["mert_frames"])
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
        entries.append((source_start, max(source_start, source_end), source_start - logical_start,
                        window_start, target_start, len(tokens), len(row), loss_start))
        tokens.extend(row)
        target_start += _PREFLIGHT_CONFIG["stride_sec"]
    return entries, tokens


def _compile(
    charts_dir: Path,
    cache_dir: Path,
    build_path: Path,
    config: dict,
    sources: list[dict],
    input_sources: list[dict],
    reusable: dict[str, tuple[dict, np.ndarray, np.ndarray]],
    parsed: dict[str, tuple[tuple[list, dict | None, set[str]] | None, str | None]],
    compiled: dict[str, tuple[list, list[int]]],
) -> None:
    entries = []
    tokens: list[int] = []
    charts = []
    excluded = 0
    rejected = []
    oversized = []
    recoveries: Counter[str] = Counter()
    parse_errors = []
    reused = 0
    for source in tqdm(sources, desc="[chart-cache] 索引", unit="首"):
        reused_chart = reusable.get(source["chart"])
        if reused_chart is not None:
            max_length = max((int(row["token_length"]) for row in reused_chart[1]), default=0)
            if max_length > config["max_tokens"]:
                oversized.append({"chart": source["chart"], "max_tokens": max_length})
                continue
            chart_id = len(charts)
            charts.append(reused_chart[0])
            _append_reused_chart(reused_chart[0], reused_chart[1], reused_chart[2], chart_id, entries, tokens)
            reused += 1
            continue
        try:
            result, error = parsed[source["chart"]]
        except KeyError:
            result, error = None, "未取得解析结果"
        if result is None:
            parse_errors.append({"chart": source["chart"], "error": error})
            continue
        _levels, song, warnings = result
        recoveries.update(warnings)
        if song is not None and song.get("basic_info", {}).get("genre") == "宴会場":
            excluded += 1
            continue
        chart_id = len(charts)
        charts.append({"chart": source["chart"], "mel": source["mel"]})
        local_entries, local_tokens = compiled[source["chart"]]
        for source_start, source_end, left_pad, window_start, target_start, token_start, token_length, loss_start in local_entries:
            entries.append((chart_id, source_start, source_end, left_pad, window_start, target_start,
                            len(tokens) + token_start, token_length, loss_start))
        tokens.extend(local_tokens)
    build_path.mkdir(parents=True)
    np.save(build_path / "entries.npy", np.array(entries, dtype=ENTRY_DTYPE))
    np.save(build_path / "tokens.npy", np.asarray(tokens, dtype=np.uint16))
    manifest = {
        "config": config,
        "input_sources": input_sources,
        "sources": sources,
        "charts": charts,
        "windows": len(entries),
        "rejected": rejected,
        "oversized": oversized,
        "parse_errors": parse_errors,
    }
    (build_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    (build_path / "COMPLETE").touch()
    print(
        f"[chart-cache] 完成: {len(charts)} 首，{len(entries)} 窗口；"
        f"复用 {reused} 首，编译 {len(sources) - reused} 首，"
        f"宴会場排除 {excluded} 首，严格规则排除 {len(rejected)} 首，"
        f"超长排除 {len(oversized)} 首，解析失败 {len(parse_errors)} 首"
    )
    if recoveries:
        labels = {
            "empty-each": "空 EACH 分支",
            "tap-duration": "TAP 孤立时长",
            "missing-slide-shape": "Slide 缺少形状",
            "extra-brace": "多余右花括号",
        }
        detail = "，".join(f"{labels.get(kind, kind)} {count} 首" for kind, count in sorted(recoveries.items()))
        print(f"[chart-cache] 兼容修复汇总: {detail}")


def ensure_chart_cache(
    charts_dir: str | Path,
    cache_dir: str | Path,
    *,
    level_idx: int,
    stride_sec: float,
    mert_frames: int,
    build_mel: bool = True,
) -> Path:
    charts_dir = Path(charts_dir)
    cache_dir = Path(cache_dir)
    config = _config(level_idx, stride_sec, mert_frames, music_data_version())
    cache_root, lock_path = _paths(cache_dir, config)
    current_path = _current_path(cache_root)
    prior_path = current_path or _find_prior_cache(cache_dir, config)
    input_sources = _scan_chart_sources(charts_dir, cache_dir)
    if current_path is not None and _is_current(current_path, config, input_sources):
        return current_path

    # 先在 CPU 预检谱面，确保不为解析失败、规则违规或超长歌曲写入 MERT。
    chart_sources = input_sources
    songs = load_music_data()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=CONFIG.mert.preflight_workers,
        initializer=_init_preflight,
        initargs=(songs, config),
    ) as executor, tqdm(total=len(chart_sources), desc="[chart-cache] 预检", unit="首") as progress:
        futures = {
            executor.submit(_parse_source, charts_dir, source): source["chart"]
            for source in chart_sources
        }
        parsed = {}
        for future in concurrent.futures.as_completed(futures):
            chart = futures[future]
            try:
                result, error = future.result()
            except Exception as exc:
                result, error = None, f"预检进程异常: {exc}"
            if error is not None:
                tqdm.write(f"[chart-cache] 错误: 跳过 {chart}: {error}")
            parsed[chart] = (result, error)
            progress.update()
        eligible_charts = {
            source["chart"]
            for source in chart_sources
            if parsed[source["chart"]][0] is not None
        }
        if build_mel:
            allowed_rel_paths = {
                str(Path(source["chart"]).parent)
                for source in chart_sources
                if source["chart"] in eligible_charts
            }
            rebuild_mert_cache(charts_dir, cache_dir, allowed_rel_paths=allowed_rel_paths)

    _acquire_lock(lock_path)
    try:
        sources = _scan_sources(charts_dir, cache_dir, eligible_charts)
        current_path = _current_path(cache_root)
        if current_path is not None and _is_current(current_path, config, input_sources):
            return current_path
        invalid_sources = [source for source in sources if not isinstance(source, dict) or "chart" not in source]
        if invalid_sources:
            for source in invalid_sources:
                tqdm.write(f"[chart-cache] 错误: 跳过异常索引来源: {source!r}")
            sources = [source for source in sources if isinstance(source, dict) and "chart" in source]
        reusable = _load_reusable(current_path or prior_path, config, sources)
        compile_sources = [source for source in sources if source["chart"] not in reusable]
        compiled = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CONFIG.mert.preflight_workers,
            initializer=_init_preflight,
            initargs=(songs, config),
        ) as executor, tqdm(total=len(compile_sources), desc="[chart-cache] 编译", unit="首") as progress:
            futures = {
                executor.submit(_compile_source, cache_dir, source, parsed[source["chart"]][0]): source["chart"]
                for source in compile_sources
            }
            for future in concurrent.futures.as_completed(futures):
                chart = futures[future]
                try:
                    compiled[chart] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"窗口编译失败: {chart}: {exc}") from exc
                progress.update()
        temp_path = cache_root / f".build-{os.getpid()}-{time.time_ns()}"
        shutil.rmtree(temp_path, ignore_errors=True)
        _compile(charts_dir, cache_dir, temp_path, config, sources, input_sources, reusable, parsed, compiled)
        _publish(cache_root, temp_path)
        return _current_path(cache_root)
    finally:
        lock_path.unlink(missing_ok=True)


def require_chart_cache(
    charts_dir: str | Path,
    cache_dir: str | Path,
    *,
    level_idx: int,
    stride_sec: float,
    mert_frames: int,
) -> Path:
    """训练只读取已发布索引，绝不在 CUDA 进程内构建缓存。"""
    charts_dir = Path(charts_dir)
    cache_dir = Path(cache_dir)
    config = _config(level_idx, stride_sec, mert_frames, music_data_version())
    cache_root, _lock_path = _paths(cache_dir, config)
    path = _current_path(cache_root)
    if path is None:
        raise FileNotFoundError(
            f"缺少索引缓存 {_key(config)}；请先运行 uv run src/cache_all.py"
        )
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("索引配置已过期")
        if manifest.get("input_sources") != _scan_chart_sources(charts_dir, cache_dir):
            raise ValueError("歌曲或谱面已变化")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"索引缓存损坏: {path}") from error
    except ValueError as error:
        raise RuntimeError(f"索引缓存过期: {error}；请先运行 uv run src/cache_all.py") from error
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", type=Path, default=CONFIG.paths.charts_dir)
    parser.add_argument("--cache-dir", type=Path, default=CONFIG.paths.mert_cache_dir)
    parser.add_argument("--level", type=_level_arg, default=CONFIG.training.level_idx)
    parser.add_argument("--stride-sec", type=_positive_float_arg, default=CONFIG.window.train_stride_sec)
    args = parser.parse_args()
    path = ensure_chart_cache(
        args.charts_dir,
        args.cache_dir,
        level_idx=args.level,
        stride_sec=args.stride_sec,
        mert_frames=CONFIG.window.mert_frames,
    )


if __name__ == "__main__":
    main()
