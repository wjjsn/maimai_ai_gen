import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


# ── Note ──────────────────────────────────────────────────────────────────


class NoteType(Enum):
    TAP = auto()
    HOLD = auto()
    SLIDE = auto()
    TOUCH = auto()
    TOUCH_HOLD = auto()


class TouchType(Enum):
    A1 = auto()
    A2 = auto()
    A3 = auto()
    A4 = auto()
    A5 = auto()
    A6 = auto()
    A7 = auto()
    A8 = auto()
    B1 = auto()
    B2 = auto()
    B3 = auto()
    B4 = auto()
    B5 = auto()
    B6 = auto()
    B7 = auto()
    B8 = auto()
    C = auto()
    D1 = auto()
    D2 = auto()
    D3 = auto()
    D4 = auto()
    D5 = auto()
    D6 = auto()
    D7 = auto()
    D8 = auto()
    E1 = auto()
    E2 = auto()
    E3 = auto()
    E4 = auto()
    E5 = auto()
    E6 = auto()
    E7 = auto()
    E8 = auto()


class TapType(Enum):
    LANE1 = auto()
    LANE2 = auto()
    LANE3 = auto()
    LANE4 = auto()
    LANE5 = auto()
    LANE6 = auto()
    LANE7 = auto()
    LANE8 = auto()


class SlideShape(Enum):
    Line = auto()  # -
    Circle = auto()  # > < ^
    V = auto()  # v
    GrandV = auto()  # V (L型折线)
    P = auto()  # p (单弯)
    Q = auto()  # q (单弯镜像)
    PP = auto()  # pp (大弯)
    QQ = auto()  # qq (大弯镜像)
    S = auto()  # s (闪电)
    Z = auto()  # z (闪电镜像)
    Wifi = auto()  # w (扇形/WiFi)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclass
class SlideSegment:
    shape: SlideShape
    start_lane: TapType
    end_lane: TapType
    wait_duration: int  # 等待时间帧数，默认一拍
    trace_duration: int  # 持续时间帧数

    isClockwise: bool | None = None  # > 顺时针 / < 逆时针 / ^ 自动判断选最短，不需要这个字段
    middle_lane: TapType | None = None  # GrandV的拐点。1V35  →  起点=1, 拐点=3, 终点=5

    isForceStar: bool = False  # 强制星星头
    isFakeRotate: bool = False  # 假旋转 ($$)
    isSlideBreak: bool = False  # Slide 本体 Break。b 在 ] 后 = 整条 Slide 变 Break
    isSlideNoHead: bool = False  # Slide 无头 (! 或 ? 或 *)


@dataclass
class Touch_data:
    Touch_area: TouchType
    isFirework: bool = False  # 烟花
    holdTime: int | None = None  # 持续时间帧数，TouchHold 才有


@dataclass
class Hold_data:
    lane: TapType
    holdTime: int  # 持续时间帧数


@dataclass
class Note:
    """一个音符。"""
    type: NoteType
    data: TapType | Hold_data | Touch_data | list[SlideSegment]
    isBreak: bool = False
    isEx: bool = False


@dataclass
class Frame:
    """一个时间帧，包含该时间点的所有音符。"""
    frame_idx: int
    notes: tuple[Note, ...] = ()
    note_tokens: tuple[str, ...] = ()


@dataclass
class Level:
    """一个难度等级，包含该难度的所有时间帧。"""
    level_name: str
    level_query: float | None
    frames: list[Frame] = field(default_factory=list)


@dataclass
class Chart:
    """一首歌的+不同难度（0.0~15.0）谱面。"""
    all_levels: list[Level | None] = field(default_factory=list)
    title: str = "default"
    artist: str = "default"
    designer: str = "default"

    # EASY/BASIC/ADVANCED/EXPERT/MASTER/Re:MASTER/ORIGINAL
    # lv_1/lv_2 / lv_3   /  lv_4/  lv_5/     lv_6/  lv_7

# 更改架构，解析器和生成器使用compile类的不同方法，均基于compile类内部的Chart对象
# 生成器使用12000BPM，通过填充','来确保解析后的数据序列化后仍然能与原始数据在时间上对齐
# $$Time\ per\ frame = \frac{hop\_length}{sample\_rate}$$

class compiler:
    def __init__(self, hop_length, sample_rate):
        self.chart = Chart(all_levels=[])
        self.time_pre_frame = hop_length / sample_rate
        self.current_time = 0.0  # seconds

    def _parse_note(self, token: str) -> Note | None:
        cleaned = re.sub(r"\([^)]*\)", "", token)
        cleaned = re.sub(r"\{[^}]*\}", "", cleaned).strip()
        if not cleaned or cleaned == "E":
            return None
        return Note(type=NoteType.TAP, data=TapType.LANE1)

    def _parse_current_time_per_comma(self, note: str, current_bpm: float, current_length_divider: float, current_per_comma_length: float) -> tuple[float, float, float]:
        bpm_match = re.search(r"\(([^)]*)\)", note)
        length_match = re.search(r"\{([^}]*)\}", note)

        next_bpm = current_bpm
        next_divider = current_length_divider
        next_per_comma = current_per_comma_length

        if bpm_match:
            next_bpm = float(bpm_match.group(1))

        if length_match:
            raw_value = length_match.group(1).strip()
            if raw_value.startswith("#"):
                next_divider = 0.0
                next_per_comma = float(raw_value[1:])
            else:
                next_divider = float(raw_value)
                if next_bpm > 0:
                    next_per_comma = 240.0 / next_bpm / next_divider

        if bpm_match or length_match:
            return next_bpm, next_divider, next_per_comma

        if next_bpm > 0 and next_divider > 0:
            next_per_comma = 240.0 / next_bpm / next_divider

        return next_bpm, next_divider, next_per_comma

    def parse(self, text: str):
        self.chart = Chart(all_levels=[None] * 7)

        title_match = re.search(r"&title=([^\n]+)", text)
        if title_match:
            self.chart.title = title_match.group(1).strip()

        artist_match = re.search(r"&artist=([^\n]+)", text)
        if artist_match:
            self.chart.artist = artist_match.group(1).strip()

        for level in range(0, 7):
            level_match = re.search(rf"&inote_{level}=([\s\S]*?)(?=&lv_{level}=|&inote_{level + 1}=|$)", text)
            if not level_match:
                continue

            level_content = level_match.group(1)
            chart_content = level_content.split(",")
            chart_content_strip = [item.strip() for item in chart_content]

            current_bpm = 0.0
            current_length_divider = 0.0
            current_per_comma_length = 1.0
            self.current_time = 0.0
            frames_by_idx: dict[int, list[Note]] = {}
            frame_tokens_by_idx: dict[int, list[str]] = {}

            for raw_note in chart_content_strip:
                if not raw_note:
                    self.current_time += current_per_comma_length
                    continue
                if raw_note == "E":
                    break

                current_bpm, current_length_divider, current_per_comma_length = self._parse_current_time_per_comma(
                    raw_note,
                    current_bpm,
                    current_length_divider,
                    current_per_comma_length,
                )

                cleaned = re.sub(r"\([^)]*\)", "", raw_note)
                cleaned = re.sub(r"\{[^}]*\}", "", cleaned).strip()
                if not cleaned or cleaned == "E":
                    self.current_time += current_per_comma_length
                    continue

                note = self._parse_note(cleaned)
                if note is None:
                    self.current_time += current_per_comma_length
                    continue

                frame_idx = int(round(self.current_time / self.time_pre_frame))
                frames_by_idx.setdefault(frame_idx, []).append(note)
                frame_tokens_by_idx.setdefault(frame_idx, []).append(cleaned)
                self.current_time += current_per_comma_length

            frames = [
                Frame(frame_idx=frame_idx, notes=tuple(frames_by_idx[frame_idx]), note_tokens=tuple(frame_tokens_by_idx[frame_idx]))
                for frame_idx in sorted(frames_by_idx)
            ]
            level_name = f"level_{level + 1}"
            level_query_match = re.search(rf"&lv_{level}=([0-9.]+)", text)
            level_query = float(level_query_match.group(1)) if level_query_match else None
            self.chart.all_levels[level] = Level(level_name=level_name, level_query=level_query, frames=frames)

        return self.chart

    def generate(self) -> str:
        lines: list[str] = []
        lines.append(f"&title={self.chart.title}")
        lines.append(f"&artist={self.chart.artist}")
        lines.append("&first=0")

        for level_idx, level in enumerate(self.chart.all_levels):
            if level is None:
                continue

            lines.append(f"&lv_{level_idx}={level.level_query or 0}")
            lines.append(f"&inote_{level_idx}=")

            if not level.frames:
                lines.append("E")
                continue

            output_lines: list[str] = []
            current_line = ""
            prev_frame_idx: int | None = None

            for frame in sorted(level.frames, key=lambda frame: frame.frame_idx):
                if not frame.note_tokens:
                    continue

                for token_idx, token in enumerate(frame.note_tokens):
                    if not current_line:
                        current_line = f"(12000){{1}}{token},"
                    else:
                        if prev_frame_idx is not None and frame.frame_idx != prev_frame_idx:
                            gap_slots = max(0, int(round(frame.frame_idx - prev_frame_idx)) - 1)
                            current_line += "," * gap_slots
                        output_lines.append(current_line)
                        current_line = f"{token},"

                    if token_idx == len(frame.note_tokens) - 1:
                        prev_frame_idx = frame.frame_idx

            if current_line:
                output_lines.append(current_line)

            lines.extend(output_lines)
            lines.append("E")

        return "\n".join(lines)


def main():
    charts_dir = Path(__file__).resolve().parent.parent / "charts"
    tmp_dir = Path(__file__).resolve().parent.parent / "tmp"
    found = errors = 0
    for chart_dir, _dirs, files in charts_dir.walk():
        if "maidata.txt" not in files:
            continue
        rel = chart_dir.relative_to(charts_dir)
        out_path = tmp_dir / rel / "maidata.txt"
        try:
            text = (chart_dir / "maidata.txt").read_text(encoding="utf-8")
            parser = compiler(hop_length=512, sample_rate=44100)
            parser.parse(text)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(parser.generate(), encoding="utf-8")
            found += 1
            print(f"[{found}] {rel} -> {out_path}")
        except Exception as e:
            import traceback
            errors += 1
            print(f"[ERR] {rel}: {e}")
            traceback.print_exc()
    print(f"\nDone: {found} parsed, {errors} errors")


if __name__ == "__main__":
    main()