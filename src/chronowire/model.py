"""論理時間、Emission、Diagnosticの公開値を定義する。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from functools import total_ordering
from typing import Generic, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@total_ordering
@dataclass(frozen=True)
class LogicalTime:
    """整数tickと有理timebaseで論理時刻を表す。

    浮動小数の累積誤差を避け、異なるtimebase同士も正確に比較する。
    """

    ticks: int
    timebase_num: int = 1
    timebase_den: int = 1

    def __post_init__(self) -> None:
        if self.timebase_num <= 0 or self.timebase_den <= 0:
            raise ValueError("timebase numerator and denominator must be positive")

    def as_fraction(self) -> Fraction:
        """秒相当の有理数へ変換する。"""

        return Fraction(self.ticks * self.timebase_num, self.timebase_den)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LogicalTime):
            return NotImplemented
        return self.as_fraction() < other.as_fraction()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogicalTime):
            return False
        return self.as_fraction() == other.as_fraction()

    def __hash__(self) -> int:
        return hash(self.as_fraction())


@dataclass(frozen=True)
class LogicalInterval:
    """Emissionが表す半開論理時間区間`[start, end)`を表す。"""

    start: LogicalTime
    end: LogicalTime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("logical interval end must not precede start")


class Severity(StrEnum):
    """Diagnosticの重大度を表す。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Diagnostic:
    """compileまたはrun中に観測した機械可読な診断を表す。"""

    severity: Severity
    code: str
    message: str
    node_id: int | None = None
    port_id: int | None = None
    interval: LogicalInterval | None = None
    details: dict[str, object] = field(default_factory=dict)


class EmissionStatus(StrEnum):
    """出力値の利用可能性を表す。"""

    OK = "ok"
    DEGRADED = "degraded"
    INVALID = "invalid"


@dataclass(frozen=True)
class Emission(Generic[T_co]):
    """値と論理時間、品質status、診断を一体で運ぶ出力単位を表す。"""

    value: T_co
    interval: LogicalInterval
    sequence: int
    status: EmissionStatus = EmissionStatus.OK
    diagnostics: tuple[Diagnostic, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelOutputs:
    """複数output Portへ一件ずつ配送するKernel戻り値を表す。"""

    values: tuple[object, ...]


def kernel_outputs(*values: object) -> KernelOutputs:
    """複数Port用の明示KernelOutputsを生成する。

    Args:
        values: output index順の値。一件以上必要。

    Returns:
        通常tupleと区別される複数Port戻り値。

    Raises:
        ValueError: valuesが空の場合。
    """

    if not values:
        raise ValueError("kernel_outputs requires at least one value")
    return KernelOutputs(tuple(values))


@dataclass(frozen=True)
class Skip:
    """Python callableが0件のEmissionを返すことを明示するmarker。"""


@dataclass(frozen=True)
class EmitMany(Generic[T_co]):
    """Python callableが複数Emissionを返すことを明示するwrapper。"""

    values: tuple[T_co, ...]


def skip() -> Skip:
    """出力なしを表すmarkerを返す。"""

    return Skip()


def emit_many(values: Iterable[T]) -> EmitMany[T]:
    """反復可能な複数値を明示的な複数Emission wrapperへ変換する。

    Raises:
        TypeError: valuesが反復不能、または文字列・bytesの場合。
    """

    if isinstance(values, (str, bytes)):
        raise TypeError("emit_many does not accept str or bytes")
    return EmitMany(tuple(values))
