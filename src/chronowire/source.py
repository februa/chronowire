"""finite/generated Sourceの公開protocolを定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .config import Config
from .model import Emission, LogicalTime

T_co = TypeVar("T_co", covariant=True)
T_in = TypeVar("T_in", contravariant=True)


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

    generated Sourceは`is_finite=False`とし、Plan.runへdurationが必要になる。
    """

    @property
    def is_finite(self) -> bool:
        """EOFへ到達するSourceならTrueを返す。"""

        ...

    def read(self, request: SourceRequest, config: Config) -> SourceBatch[T_co]:
        """指定論理区間に対応するEmission batchを返す。"""

        ...


class RealtimeOverflowPolicy(StrEnum):
    """停止不能なRealtime Sourceのingress overflow規則。"""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


@runtime_checkable
class RealtimeReceiver(Protocol[T_in]):
    """Realtime SourceからExecutor ingressへEmissionを配送するprotocol。"""

    def publish(self, emission: Emission[T_in]) -> None:
        """一件を非blockingでingressへ配送する。"""

        ...

    def close(self) -> None:
        """正常な入力終了を通知する。"""

        ...

    def fail(self, error: BaseException) -> None:
        """入力側の回復不能な失敗を通知する。"""

        ...


@runtime_checkable
class RealtimeSourceSession(Protocol):
    """一回のrunに閉じたRealtime Source受付状態。"""

    def stop(self) -> None:
        """run終了時に外部callbackの受付を停止する。"""

        ...


@runtime_checkable
class RealtimeSource(Protocol[T_co]):
    """Schedulerが生成を停止できない外部push Sourceのprotocol。"""

    @property
    def max_items(self) -> int:
        """一時的な処理時間の揺らぎを吸収するingress件数を返す。"""

        ...

    @property
    def overflow_policy(self) -> RealtimeOverflowPolicy:
        """ingress満杯時の明示的な棄却規則を返す。"""

        ...

    def start(
        self,
        receiver: RealtimeReceiver[T_co],
        config: Config,
    ) -> RealtimeSourceSession:
        """run-local receiverへ配送を開始し、停止用sessionを返す。"""

        ...
