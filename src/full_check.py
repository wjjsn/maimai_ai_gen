"""扫描全量谱面，按现有格式生成统一 12000 BPM 文本。"""

import shutil
from pathlib import Path

from config import CONFIG, ROOT_DIR
from maidata_parser import generate_maidata, parse_maidata


OUTPUT_DIR = ROOT_DIR / "tmp" / "maidata-12000"


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    converted = 0
    errors: list[str] = []
    for source in sorted(CONFIG.paths.charts_dir.rglob("maidata.txt")):
        relative = source.relative_to(CONFIG.paths.charts_dir)
        try:
            chart = parse_maidata(source.read_text(encoding="utf-8"))
            text = generate_maidata(chart)
            parse_maidata(text)
            destination = OUTPUT_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            converted += 1
        except Exception as error:
            errors.append(f"{relative}: {error}")
    print(f"[full-check] 转换成功={converted} 转换失败={len(errors)} 输出={OUTPUT_DIR}")
    for error in errors[:20]:
        print(f"  {error}")
    if errors:
        raise RuntimeError(f"全量谱面检查失败 {len(errors)} 首")


if __name__ == "__main__":
    main()
