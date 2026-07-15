"""下载并匹配 Diving-Fish 的 maimai 精确难度定数。"""

import json
import hashlib
import math
import os
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

from config import ROOT_DIR


MUSIC_DATA_URL = "https://www.diving-fish.com/api/maimaidxprober/music_data"
CACHE_PATH = ROOT_DIR / ".cache" / "diving-fish-music-data.json"
DATA_SCHEMA_VERSION = 1


def _validate_music_data(data) -> list[dict]:
    if not isinstance(data, list) or not data:
        raise ValueError("歌曲数据必须是非空数组")
    for index, song in enumerate(data):
        if not isinstance(song, dict) or not all(key in song for key in ("id", "title", "type", "ds")):
            raise ValueError(f"第 {index} 首歌曲缺少必要字段")
        if not isinstance(song["title"], str) or song["type"] not in ("SD", "DX") or not isinstance(song["ds"], list):
            raise ValueError(f"第 {index} 首歌曲字段类型错误")
        if not 1 <= len(song["ds"]) <= 5:
            raise ValueError(f"第 {index} 首歌曲难度数量错误")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in song["ds"]):
            raise ValueError(f"第 {index} 首歌曲含无效定数")
    return data


def _serialize(data: list[dict]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def music_data_digest(data: list[dict]) -> str:
    return hashlib.sha256(_serialize(data).encode()).hexdigest()


def load_music_data(refresh: bool = False) -> tuple[list[dict], str]:
    if CACHE_PATH.is_file() and not refresh:
        try:
            data = _validate_music_data(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
            return data, music_data_digest(data)
        except Exception as error:
            raise RuntimeError(f"Diving-Fish 本地缓存损坏，请删除后重新下载：{error}") from error

    try:
        with urlopen(MUSIC_DATA_URL, timeout=30) as response:
            data = _validate_music_data(json.load(response))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_PATH.with_suffix(".tmp")
        temporary.write_text(_serialize(data), encoding="utf-8")
        os.replace(temporary, CACHE_PATH)
        return data, music_data_digest(data)
    except Exception as error:
        raise RuntimeError(f"无法获取 Diving-Fish 歌曲数据：{error}") from error


def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title).strip()
    title = re.sub(r"\s*\[(?:DX|ST)\]\s*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title)


def match_music(text: str, chart_path: Path, music_data: list[dict]) -> dict | None:
    title_match = re.search(r"^&title=([^\r\n]+)", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else chart_path.parent.name
    if chart_path.parent.name.startswith("[") or title.startswith("["):
        return None

    normalized = normalize_title(title)
    candidates = [song for song in music_data if normalize_title(song["title"]) == normalized]
    suffix = re.search(r"\[(DX|ST)\]\s*$", title, re.IGNORECASE)
    if suffix:
        expected_type = suffix.group(1).upper().replace("ST", "SD")
        candidates = [song for song in candidates if song["type"] == expected_type]

    shortid = re.search(r"^&shortid=(\d+)", text, re.MULTILINE)
    if shortid:
        matched = [song for song in candidates if str(song["id"]) == shortid.group(1)]
        return matched[0] if len(matched) == 1 else None

    folder_id = re.match(r"(\d+)_", chart_path.parent.name)
    if folder_id:
        matched = [song for song in candidates if str(song["id"]) == folder_id.group(1)]
        if len(matched) == 1:
            return matched[0]

    if not suffix:
        standard = [song for song in candidates if song["type"] == "SD"]
        if len(standard) == 1:
            return standard[0]
    return candidates[0] if len(candidates) == 1 else None


def _self_check() -> None:
    data = [
        {"id": "1", "title": "same", "type": "SD"},
        {"id": "10001", "title": "same", "type": "DX"},
    ]
    assert match_music("&title=same", Path("same/maidata.txt"), data)["id"] == "1"
    assert match_music("&title=same [DX]", Path("same/maidata.txt"), data)["id"] == "10001"
    assert match_music("&title=other\n&shortid=1", Path("other/maidata.txt"), data) is None
    assert match_music("&title=same [DX]\n&shortid=1", Path("same/maidata.txt"), data) is None
    ambiguous = data + [{"id": "2", "title": "same", "type": "SD"}]
    assert match_music("&title=same", Path("same/maidata.txt"), ambiguous) is None
    assert match_music("&title=same", Path("1_same/maidata.txt"), ambiguous)["id"] == "1"
    assert match_music("&title=[宴] same", Path("[宴] same/maidata.txt"), data) is None
    assert len(music_data_digest(_validate_music_data([{**data[0], "ds": [1.0]}]))) == 64
    print("[music-data] 自检通过")


if __name__ == "__main__":
    _self_check()
