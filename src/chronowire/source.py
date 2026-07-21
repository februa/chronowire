"""finite/generated Sourceの公開protocolを定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .config import Config
from .model import Emission, LogicalTime

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class SourceRequest:
    """SchedulerがSourceへ要求する論理時間区間を表す。"""

    logical_start: LogicalTime
    duration: Fraction


@dataclass(frozen=True)
class SourceBatch(Generic[T_co]):
    """Sourceが一回のrequestで返すEmission列とEOF状態を表す。"""

    emissions: tuple[Emission[T_co], ...]
    eof: bool = False


@runtime_checkable
class Source(Protocol[T_co]):
    """論理時間requestに応じてEmissionを供給する公開protocol。

    generated Sourceは`is_finite=False`とし、ExecutionPlan.runへdurationが必要になる。
    """

    @property
    def is_finite(self) -> bool:
        """EOFへ到達するSourceならTrueを返す。"""

        ...

    def read(self, request: SourceRequest, config: Config) -> SourceBatch[T_co]:
        """指定論理区間に対応するEmission batchを返す。"""

        ...
