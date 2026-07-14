from dataclasses import dataclass, field
from enum import Enum, auto


class NoteType(Enum):
    TAP = auto()
    HOLD = auto()
    SLIDE = auto()
    TOUCH = auto()
    TOUCH_HOLD = auto()


class TouchType(Enum):
    A1 = auto(); A2 = auto(); A3 = auto(); A4 = auto(); A5 = auto(); A6 = auto(); A7 = auto(); A8 = auto()
    B1 = auto(); B2 = auto(); B3 = auto(); B4 = auto(); B5 = auto(); B6 = auto(); B7 = auto(); B8 = auto()
    C = auto()
    D1 = auto(); D2 = auto(); D3 = auto(); D4 = auto(); D5 = auto(); D6 = auto(); D7 = auto(); D8 = auto()
    E1 = auto(); E2 = auto(); E3 = auto(); E4 = auto(); E5 = auto(); E6 = auto(); E7 = auto(); E8 = auto()


class TapType(Enum):
    LANE1 = auto(); LANE2 = auto(); LANE3 = auto(); LANE4 = auto()
    LANE5 = auto(); LANE6 = auto(); LANE7 = auto(); LANE8 = auto()


class SlideShape(Enum):
    Line = auto()
    Circle = auto()
    V = auto()
    GrandV = auto()
    P = auto()
    Q = auto()
    PP = auto()
    QQ = auto()
    S = auto()
    Z = auto()
    Wifi = auto()


@dataclass
class SlideSegment:
    shape: SlideShape
    start_lane: TapType
    end_lane: TapType
    wait_duration: float = 0.0
    trace_duration: float = 0.0
    is_default_wait: bool = True
    isClockwise: bool | None = None
    middle_lane: TapType | None = None
    isForceStar: bool = False
    isFakeRotate: bool = False
    isSlideBreak: bool = False
    isSlideNoHead: bool = False


@dataclass
class TouchData:
    Touch_area: TouchType
    isFirework: bool = False
    holdTime: float = 0.0


@dataclass
class HoldData:
    lane: TapType
    holdTime: float = 0.0


@dataclass
class Note:
    type: NoteType
    data: TapType | HoldData | TouchData | list[SlideSegment]
    isBreak: bool = False
    isEx: bool = False


@dataclass
class Frame:
    notes: tuple[Note, ...] = ()
    time_sec: float = 0.0


@dataclass
class Level:
    level_name: str
    level_query: float | None
    frames: list[Frame] = field(default_factory=list)


@dataclass
class Chart:
    all_levels: list[Level | None] = field(default_factory=lambda: [None] * 7)
    title: str = "default"
    artist: str = "default"
    designer: str = "default"
    first_sec: float = 0.0
