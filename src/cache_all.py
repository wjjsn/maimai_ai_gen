"""在独立 CPU 进程中预检谱面、补齐 MERT 并构建训练索引。"""

from config import CONFIG
from chart_cache import ensure_chart_cache


def main() -> None:
    for stride_sec in (CONFIG.window.train_stride_sec, CONFIG.window.infer_stride_sec):
        ensure_chart_cache(
            CONFIG.paths.charts_dir,
            CONFIG.paths.mert_cache_dir,
            level_idx=CONFIG.training.level_idx,
            stride_sec=stride_sec,
            mert_frames=CONFIG.window.mert_frames,
        )


if __name__ == "__main__":
    main()
