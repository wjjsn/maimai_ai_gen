"""解析 maidata.txt 谱面部分，写回 ./tmp/{原路径}。

数据结构:
  Note    — 一个音符，含 dt (与前一个音符的时间差，ms)
  Chart   — 一首歌的一个难度谱面 + 难度等级
  MaidataFile — 一首歌的 metadata + 各难度 Chart

序列化时从 dt 重建 BPM/逗号，格式可能与原文不同，但时间和内容一致。
"""

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


@dataclass
class Note:
    """一个音符。dt 是与前一个 Note 的时间差 (ms)。"""
    type: NoteType
    dt: float                              # Δt from previous note, ms
    lane: int | str | None = None          # 按钮 1-8 / 触摸区 "B1"/"C"

    # 时值 [num:den]，秒数时 den=None
    duration_num: int | float | None = None
    duration_den: int | None = None

    # SLIDE
    slide_shapes: list[str] | None = None
    slide_end: int | None = None
    slide_wait: str | None = None
    slide_duration_num: int | float | None = None
    slide_duration_den: int | None = None

    # 标记
    break_: bool = False
    ex: bool = False
    star: str | None = None
    firework: bool = False
    hidden_star: str | None = None
    pseudo_each: bool = False


@dataclass
class Chart:
    """一首歌的+不同难度（0.0~15.0）谱面。"""

    east: list[tuple[Note, ...]] | None = None
    easy_difficulty_level: str | None = None
    basic: list[tuple[Note, ...]] | None = None
    basic_difficulty_level: str | None = None
    advanced: list[tuple[Note, ...]] | None = None
    advanced_difficulty_level: str | None = None
    expert: list[tuple[Note, ...]] | None = None
    expert_difficulty_level: str | None = None
    master: list[tuple[Note, ...]] | None = None
    master_difficulty_level: str | None = None
    remaster: list[tuple[Note, ...]] | None = None
    remaster_difficulty_level: str | None = None


# ── 解析器 ────────────────────────────────────────────────────────────────

_RE_META = re.compile(r"^&(\w+)=(.*)$")
_RE_CHART_HEADER = re.compile(r"^&(inote_\w+)=(.*)$")

_DIFF_KEY_MAP = {
    "2": ("basic", "basic_difficulty_level"),
    "3": ("advanced", "advanced_difficulty_level"),
    "4": ("expert", "expert_difficulty_level"),
    "5": ("master", "master_difficulty_level"),
    "1": ("east", "easy_difficulty_level"),
    "0": ("remaster", "remaster_difficulty_level"),
}


class Parser:
    """解析 simai 谱面文本，返回 Note 列表 (只有 dt，无 TIMING)。"""

    __slots__ = ("text", "pos", "len", "bpm", "ms_per_div", "elapsed_ms", "prev_note_abs")

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.len = len(text)
        self.bpm = 0.0
        self.ms_per_div = 0.0
        self.elapsed_ms = 0.0
        self.prev_note_abs = 0.0

    def parse(self) -> list[Note]:
        notes: list[Note] = []
        while self.pos < self.len:
            self._skip_ws()
            if self.pos >= self.len:
                break
            c = self.text[self.pos]
            if c == ",":
                self.elapsed_ms += self.ms_per_div
                self.pos += 1
            elif c == "(":
                self.bpm = self._parse_bpm()
                self._update_ms_per_div()
            elif c == "{":
                self._apply_time_div(self._parse_braced())
            else:
                self._parse_note_group(notes)
        return notes

    def _skip_ws(self):
        while self.pos < self.len and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _update_ms_per_div(self):
        if self.bpm > 0:
            self.ms_per_div = 240_000.0 / self.bpm

    def _apply_time_div(self, div_str: str):
        if div_str.startswith("#"):
            self.ms_per_div = float(div_str[1:]) * 1000
        elif self.bpm > 0:
            self.ms_per_div = 240_000.0 / self.bpm / float(div_str)

    def _parse_bpm(self) -> float:
        self.pos += 1
        start = self.pos
        while self.pos < self.len and self.text[self.pos] != ")":
            self.pos += 1
        bpm = float(self.text[start:self.pos])
        if self.pos < self.len:
            self.pos += 1
        return bpm

    def _parse_braced(self) -> str:
        self.pos += 1
        start = self.pos
        while self.pos < self.len and self.text[self.pos] != "}":
            self.pos += 1
        s = self.text[start:self.pos]
        if self.pos < self.len:
            self.pos += 1
        return s

    # ── 音符解析 ──────────────────────────────────────────────────────

    def _parse_note_group(self, notes: list[Note]):
        """解析逗号前的所有音符。"""
        token_buf: list[str] = []
        while self.pos < self.len:
            c = self.text[self.pos]
            if c in ",(){}":
                break
            token_buf.append(c)
            self.pos += 1

        token = "".join(token_buf).strip()
        if not token:
            return

        pe_groups = token.split("`")
        for group in pe_groups:
            group = group.strip()
            if not group:
                continue
            is_pe = len(pe_groups) > 1
            if is_pe:
                self.elapsed_ms += 1

            for part in group.split("/"):
                part = part.strip()
                if not part:
                    continue
                for note in self._parse_token(part):
                    note.pseudo_each = is_pe
                    note.dt = self.elapsed_ms - self.prev_note_abs
                    self.prev_note_abs = self.elapsed_ms
                    notes.append(note)

    def _parse_token(self, token: str) -> list[Note]:
        if not token:
            return []
        c = token[0]
        if c in "ABCDEabcde":
            return [self._parse_touch(token)]
        if c.isdigit() and len(token) > 1 and all(ch.isdigit() for ch in token):
            return [self._make_tap(int(ch)) for ch in token]
        return [self._parse_single_note(token)]

    def _make_tap(self, lane: int) -> Note:
        return Note(type=NoteType.TAP, dt=0.0, lane=lane)

    def _parse_single_note(self, token: str) -> Note:
        if token[0] in "ABCDEabcde":
            return self._parse_touch(token)
        return self._parse_button(token)

    # ── 按钮音符 ──────────────────────────────────────────────────────

    def _parse_button(self, token: str) -> Note:
        pos = 0
        tlen = len(token)
        lane, pos = self._parse_lane(token, pos)

        star = hidden = None
        while pos < tlen and token[pos] in "$@?!":
            if token[pos] in "$@":
                star = token[pos]
            else:
                hidden = token[pos]
            pos += 1

        if pos < tlen and token[pos] in "hH":
            return self._parse_hold(token, pos, lane, star)
        if pos < tlen and self._is_slide_char(token[pos]):
            return self._parse_slide(token, pos, lane, star, hidden)

        break_, ex = self._parse_bx(token, pos)
        return Note(
            type=NoteType.TAP, dt=0.0, lane=lane,
            break_=break_, ex=ex, star=star, hidden_star=hidden,
        )

    def _parse_lane(self, token: str, pos: int) -> tuple[int, int]:
        start = pos
        while pos < len(token) and token[pos].isdigit():
            pos += 1
        if start == pos:
            raise ValueError(f"Expected digit at pos {pos} in '{token}'")
        return int(token[start:pos]), pos

    @staticmethod
    def _is_slide_char(c: str) -> bool:
        return c in "-<>^vpqszVw"

    def _parse_bx(self, token: str, pos: int) -> tuple[bool, bool]:
        break_ = ex = False
        while pos < len(token) and token[pos] in "bBxX":
            if token[pos] in "bB":
                break_ = True
            else:
                ex = True
            pos += 1
        return break_, ex

    # ── HOLD ──────────────────────────────────────────────────────────

    def _parse_hold(self, token: str, pos: int, lane: int, star: str | None) -> Note:
        pos += 1
        break_, ex = self._parse_bx(token, pos)
        while pos < len(token) and token[pos] in "bBxX":
            pos += 1
        num = den = None
        if pos < len(token) and token[pos] == "[":
            num, den, pos = self._parse_duration(token, pos)
        return Note(
            type=NoteType.HOLD, dt=0.0, lane=lane,
            duration_num=num, duration_den=den,
            break_=break_, ex=ex, star=star,
        )

    # ── SLIDE ─────────────────────────────────────────────────────────

    def _parse_slide(self, token: str, pos: int, start_lane: int,
                     star: str | None, hidden: str | None) -> Note:
        shapes: list[str] = []
        wait = dur_num = dur_den = end_lane = None
        tlen = len(token)

        shape, pos = self._parse_slide_shape(token, pos)
        shapes.append(shape)
        end_lane, pos = self._parse_slide_end(token, pos)
        dur_num, dur_den, wait, pos = self._parse_slide_duration(token, pos)

        while pos < tlen and self._is_slide_char(token[pos]):
            shape, pos = self._parse_slide_shape(token, pos)
            shapes.append(shape)
            end_lane, pos = self._parse_slide_end(token, pos)
            d_n, d_d, w, pos = self._parse_slide_duration(token, pos)
            if d_n is not None:
                dur_num, dur_den, wait = d_n, d_d, w

        break_ = pos < tlen and token[pos] in "bB"
        return Note(
            type=NoteType.SLIDE, dt=0.0, lane=start_lane,
            slide_shapes=shapes, slide_end=end_lane,
            slide_wait=wait,
            slide_duration_num=dur_num, slide_duration_den=dur_den,
            break_=break_, star=star, hidden_star=hidden,
        )

    def _parse_slide_shape(self, token: str, pos: int) -> tuple[str, int]:
        if pos + 1 < len(token) and token[pos:pos+2] in ("pp", "qq"):
            return token[pos:pos+2], pos + 2
        return token[pos], pos + 1

    def _parse_slide_end(self, token: str, pos: int) -> tuple[int | None, int]:
        if pos < len(token) and token[pos].isdigit():
            return self._parse_lane(token, pos)
        return None, pos

    def _parse_slide_duration(self, token: str, pos: int
                              ) -> tuple[int | float | None, int | None, str | None, int]:
        tlen = len(token)
        if pos >= tlen or token[pos] != "[":
            return None, None, None, pos
        pos += 1
        start = pos
        while pos < tlen and token[pos] != "]":
            pos += 1
        inner = token[start:pos]
        if pos < tlen:
            pos += 1

        wait = num = den = None
        hashes = [i for i, c in enumerate(inner) if c == "#"]

        if len(hashes) == 0:
            num, den = self._parse_ratio(inner)
        elif len(hashes) == 1:
            h = hashes[0]
            if h == 0:
                rest = inner[1:]
                if ":" in rest:
                    num, den = self._parse_ratio(rest)
                elif rest:
                    num = self._to_number(rest)
            elif inner.endswith("#"):
                wait = inner[:h]
            else:
                wait = inner[:h]
                rest = inner[h+1:]
                if ":" in rest:
                    num, den = self._parse_ratio(rest)
                elif rest:
                    num = self._to_number(rest)
        elif len(hashes) == 2:
            h2 = hashes[1]
            wait = inner[:hashes[0]]
            rest = inner[h2+1:]
            if ":" in rest:
                num, den = self._parse_ratio(rest)
            elif rest:
                num = self._to_number(rest)

        return num, den, wait, pos

    # ── TOUCH ─────────────────────────────────────────────────────────

    def _parse_touch(self, token: str) -> Note:
        pos = 0
        tlen = len(token)
        group = token[pos].upper()
        pos += 1

        num = None
        if pos < tlen and token[pos].isdigit():
            start = pos
            while pos < tlen and token[pos].isdigit():
                pos += 1
            num = int(token[start:pos])
        lane = group if num is None else f"{group}{num}"

        is_hold = firework = False

        if pos < tlen and token[pos] == "f":
            firework = True
            pos += 1
        if pos < tlen and token[pos] in "hH":
            is_hold = True
            pos += 1
            if pos < tlen and token[pos] == "f":
                firework = True
                pos += 1
        elif pos < tlen and token[pos] == "f":
            firework = True
            pos += 1

        break_, ex = self._parse_bx(token, pos)
        while pos < tlen and token[pos] in "bBxX":
            pos += 1

        num_val = den_val = None
        if is_hold:
            if pos < tlen and token[pos] == "[":
                num_val, den_val, pos = self._parse_duration(token, pos)
            else:
                num_val, den_val = 1280, 1

        ntype = NoteType.TOUCH_HOLD if is_hold else NoteType.TOUCH
        return Note(
            type=ntype, dt=0.0, lane=lane,
            duration_num=num_val, duration_den=den_val,
            break_=break_, ex=ex, firework=firework,
        )

    # ── duration ──────────────────────────────────────────────────────

    def _parse_duration(self, token: str, pos: int
                        ) -> tuple[int | float | None, int | None, int]:
        tlen = len(token)
        if token[pos] != "[":
            return None, None, pos
        pos += 1
        start = pos
        while pos < tlen and token[pos] != "]":
            pos += 1
        inner = token[start:pos]
        if pos < tlen:
            pos += 1
        if ":" in inner:
            return (*self._parse_ratio(inner), pos)
        return (self._to_number(inner) if inner else None, None, pos)

    @staticmethod
    def _parse_ratio(s: str) -> tuple[int | None, int | None]:
        if ":" in s:
            parts = s.split(":", 1)
            return (int(parts[0]) if parts[0] else None,
                    int(parts[1]) if parts[1] else None)
        return (int(s) if s else None, None)

    @staticmethod
    def _to_number(s: str) -> int | float:
        return float(s) if "." in s else int(s)


# ── 序列化 ────────────────────────────────────────────────────────────────

def serialize_note(note: Note) -> str:
    if note.type in (NoteType.TOUCH, NoteType.TOUCH_HOLD):
        return _ser_touch(note)
    if note.type == NoteType.SLIDE:
        return _ser_slide(note)
    if note.type == NoteType.HOLD:
        return _ser_hold(note)
    return _ser_tap(note)


def _ser_tap(n: Note) -> str:
    s = str(n.lane)
    if n.star in ("$", "@"):
        s += n.star
    if n.hidden_star:
        s += n.hidden_star
    if n.break_:
        s += "b"
    if n.ex:
        s += "x"
    return s


def _ser_hold(n: Note) -> str:
    s = str(n.lane)
    if n.star:
        s += n.star
    s += "h"
    if n.break_:
        s += "b"
    if n.ex:
        s += "x"
    s += _fmt_dur(n.duration_num, n.duration_den)
    return s


def _ser_slide(n: Note) -> str:
    s = str(n.lane)
    if n.star in ("$", "@"):
        s += n.star
    if n.hidden_star:
        s += n.hidden_star
    for shape in (n.slide_shapes or []):
        s += shape
    if n.slide_end is not None:
        s += str(n.slide_end)
    if n.slide_wait is not None or n.slide_duration_num is not None:
        s += "["
        if n.slide_wait is not None:
            s += n.slide_wait + "#"
        if n.slide_duration_num is not None:
            if n.slide_wait is not None:
                s += "#"
            if n.slide_duration_den is not None:
                s += f"{n.slide_duration_num}:{n.slide_duration_den}"
            else:
                s += str(n.slide_duration_num)
        s += "]"
    if n.break_:
        s += "b"
    return s


def _ser_touch(n: Note) -> str:
    s = str(n.lane)
    if n.firework:
        s += "f"
    if n.type == NoteType.TOUCH_HOLD:
        s += "h"
    if n.break_:
        s += "b"
    if n.ex:
        s += "x"
    if n.type == NoteType.TOUCH_HOLD:
        s += _fmt_dur(n.duration_num, n.duration_den)
    return s


def _fmt_dur(num: int | float | None, den: int | None) -> str:
    if num is None:
        return ""
    if den is not None:
        return f"[{int(num)}:{int(den)}]"
    return f"[{num}]" if isinstance(num, float) else f"[{int(num)}]"


def _group_notes(notes: list[Note]) -> list[list[Note]]:
    """按 pseudo_each 标记分组。"""
    groups: list[list[Note]] = []
    current: list[Note] = []
    for note in notes:
        if note.pseudo_each and current and not current[0].pseudo_each:
            groups.append(current)
            current = []
        elif not note.pseudo_each and current and current[0].pseudo_each:
            groups.append(current)
            current = []
        current.append(note)
    if current:
        groups.append(current)
    return groups


def serialize_notes(notes: list[tuple[Note, ...]]) -> str:
    """从 tuple 列表重建 simai 文本。用固定 BPM=240 + {#seconds} 将音符放到正确时间点。"""
    if not notes:
        return ""

    parts: list[str] = ["(240)"]
    cursor_ms = 0.0
    abs_ms = 0.0

    for gi_idx, group in enumerate(notes):
        # 每组第一个 note 的 dt 是与前一组的时间差
        abs_ms += group[0].dt

        # 推进光标到 abs_ms
        gap = abs_ms - cursor_ms
        if gap > 0:
            full = int(gap / 1000.0)
            frac = gap - full * 1000.0
            if full > 0:
                parts.append("," * full)
            if frac > 0.01:
                parts.append(f"{{#{frac / 1000:.6f}}},(240)")
            cursor_ms = abs_ms

        # 输出音符（EACH 用 / 连接，PSEUDO_EACH 用 ` 分组）
        sub = _group_notes(list(group))
        for gi, sg in enumerate(sub):
            if gi > 0:
                parts.append("`")
            for ni, n in enumerate(sg):
                if ni > 0:
                    parts.append("/")
                parts.append(serialize_note(n))
        # 结束本组音符（逗号在下一组的 gap 中输出，或在末尾补一个）
        if gi_idx == len(notes) - 1:
            parts.append(",")

    return "".join(parts)


# ── maidata.txt ───────────────────────────────────────────────────────────

@dataclass
class MaidataFile:
    metadata: dict[str, str] = field(default_factory=dict)
    charts: dict[str, list[Note]] = field(default_factory=dict)


def parse_maidata(text: str) -> MaidataFile:
    result = MaidataFile()
    lines = text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip()
        m = _RE_CHART_HEADER.match(line)
        if m:
            key = m.group(1)
            rest = m.group(2)
            chart_lines: list[str] = []
            if rest:
                chart_lines.append(rest)
            i += 1
            while i < n:
                stripped = lines[i].rstrip()
                if re.match(r"^[ \t]*E[ \t]*$", stripped):
                    i += 1
                    break
                if _RE_CHART_HEADER.match(stripped):
                    break
                chart_lines.append(lines[i])
                i += 1
            try:
                result.charts[key] = Parser("\n".join(chart_lines)).parse()
            except Exception as e:
                print(f"[WARN] 解析 {key} 失败: {e}")
                result.charts[key] = []
        else:
            m2 = _RE_META.match(line)
            if m2:
                result.metadata[m2.group(1)] = m2.group(2)
            i += 1

    return result


def _group_by_time(notes: list[Note]) -> list[tuple[Note, ...]]:
    """将 Note 列表按绝对时间分组为 tuple 列表。"""
    groups: list[tuple[float, list[Note]]] = []
    abs_ms = 0.0
    for note in notes:
        abs_ms += note.dt
        if groups and abs(abs_ms - groups[-1][0]) < 0.5:
            groups[-1][1].append(note)
        else:
            groups.append((abs_ms, [note]))
    return [tuple(g) for _, g in groups]


def parse_maidata_to_chart(text: str) -> tuple[MaidataFile, Chart]:
    """解析 maidata.txt，返回 (MaidataFile, Chart)。"""
    mf = parse_maidata(text)
    chart = Chart()
    for key, notes in mf.charts.items():
        suffix = key.removeprefix("inote_")
        if suffix in _DIFF_KEY_MAP:
            chart_field, level_field = _DIFF_KEY_MAP[suffix]
            setattr(chart, chart_field, _group_by_time(notes))
            lv_key = f"lv_{suffix}"
            if lv_key in mf.metadata:
                setattr(chart, level_field, mf.metadata[lv_key])
    return mf, chart


def write_maidata(file: MaidataFile) -> str:
    parts: list[str] = []
    written: set[str] = set()

    for key, value in file.metadata.items():
        parts.append(f"&{key}={value}")
        if key in file.charts:
            parts.append(serialize_notes(_group_by_time(file.charts[key])))
            parts.append("E")
            written.add(key)

    for key, notes in file.charts.items():
        if key not in written:
            parts.append(f"&{key}=")
            parts.append(serialize_notes(_group_by_time(notes)))
            parts.append("E")

    return "\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────────

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
            mf = parse_maidata(text)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(write_maidata(mf), encoding="utf-8")
            found += 1
            print(f"[{found}] {rel} -> {out_path}")
        except Exception as e:
            errors += 1
            print(f"[ERR] {rel}: {e}")

    print(f"\nDone: {found} parsed, {errors} errors")


if __name__ == "__main__":
    main()
