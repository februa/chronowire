"""観測終端のboundedな値保持policyを定義する。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .model import Emission

T = TypeVar("T")


class BufferOverflowError(RuntimeError):
    """Bounded collectorがFAIL policyで上限へ達した場合の例外。"""


class OverflowPolicy(StrEnum):
    """Bounded collectorの上限到達時policyを表す。"""

    FAIL = "fail"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


@dataclass(frozen=True)
class CollectorInfo:
    """実行結果へ保存するcollector設定を表す。"""

    kind: str
    capacity: int | None = None
    overflow: OverflowPolicy | None = None


@dataclass(frozen=True)
class CollectorSnapshot(Generic[T]):
    """一回のrunでcollectorが保持した値と件数を表す。"""

    emissions: tuple[Emission[T], ...]
    received_count: int
    dropped_count: int
    info: CollectorInfo


class CollectorSession(Protocol[T]):
    """一回のrunに閉じたcollector状態のprotocol。"""

    def add(self, emission: Emission[T]) -> None:
        """Emissionを一件受け取る。"""

    def snapshot(self) -> CollectorSnapshot[T]:
        """run終了時の不変snapshotを返す。"""

        ...


@runtime_checkable
class Collector(Protocol[T]):
    """runごとに独立したCollectorSessionを生成するprotocol。"""

    def create_session(self) -> CollectorSession[T]:
        """空のrun-local sessionを生成する。"""

        ...


class _NoCollectSession(Generic[T]):
    def __init__(self) -> None:
        self._received = 0

    def add(self, emission: Emission[T]) -> None:
        self._received += 1

    def snapshot(self) -> CollectorSnapshot[T]:
        return CollectorSnapshot((), self._received, self._received, CollectorInfo("none"))


@dataclass(frozen=True)
class NoCollect(Generic[T]):
    """値を保持せず、受信件数だけを記録するcollector。"""

    def create_session(self) -> CollectorSession[T]:
        """空のNoCollect sessionを返す。"""

        return _NoCollectSession()


class _LatestSession(Generic[T]):
    def __init__(self) -> None:
        self._latest: Emission[T] | None = None
        self._received = 0

    def add(self, emission: Emission[T]) -> None:
        self._received += 1
        self._latest = emission

    def snapshot(self) -> CollectorSnapshot[T]:
        emissions = () if self._latest is None else (self._latest,)
        dropped = max(0, self._received - len(emissions))
        return CollectorSnapshot(emissions, self._received, dropped, CollectorInfo("latest", 1))


@dataclass(frozen=True)
class Latest(Generic[T]):
    """最新Emission一件だけを保持するcollector。"""

    def create_session(self) -> CollectorSession[T]:
        """空のLatest sessionを返す。"""

        return _LatestSession()


class _BoundedSession(Generic[T]):
    def __init__(self, max_items: int, overflow: OverflowPolicy) -> None:
        self._max_items = max_items
        self._overflow = overflow
        self._items: list[Emission[T]] = []
        self._received = 0
        self._dropped = 0

    def add(self, emission: Emission[T]) -> None:
        self._received += 1
        if len(self._items) < self._max_items:
            self._items.append(emission)
            return
        if self._overflow is OverflowPolicy.FAIL:
            raise BufferOverflowError(f"collector capacity {self._max_items} exceeded")
        if self._overflow is OverflowPolicy.BLOCK:
            raise BufferOverflowError(
                "BLOCK requires an asynchronous Sink and is unsupported in v0.1"
            )
        self._dropped += 1
        if self._overflow is OverflowPolicy.DROP_OLDEST:
            self._items.pop(0)
            self._items.append(emission)

    def snapshot(self) -> CollectorSnapshot[T]:
        return CollectorSnapshot(
            tuple(self._items),
            self._received,
            self._dropped,
            CollectorInfo("bounded", self._max_items, self._overflow),
        )


@dataclass(frozen=True)
class Bounded(Generic[T]):
    """最大件数を超えないcollector設定。

    BLOCKは同期runtimeでは安全に待機先を作れないため、v0.1では明示的に失敗する。
    """

    max_items: int
    overflow: OverflowPolicy = OverflowPolicy.FAIL

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("max_items must be positive")
        if self.overflow is OverflowPolicy.BLOCK:
            raise ValueError(
                "Bounded BLOCK requires a concurrent consumer and is unsupported in v0.1"
            )

    def create_session(self) -> CollectorSession[T]:
        """空のBounded sessionを返す。"""

        return _BoundedSession(self.max_items, self.overflow)


class _SinkSession(Generic[T]):
    def __init__(self, callback: Callable[[Emission[T]], None]) -> None:
        self._callback = callback
        self._received = 0

    def add(self, emission: Emission[T]) -> None:
        self._received += 1
        self._callback(emission)

    def snapshot(self) -> CollectorSnapshot[T]:
        return CollectorSnapshot((), self._received, 0, CollectorInfo("sink"))


@dataclass(frozen=True)
class Sink(Generic[T]):
    """Emissionを同期callbackへ逐次渡し、値を保持しないcollector。"""

    callback: Callable[[Emission[T]], None]

    def create_session(self) -> CollectorSession[T]:
        """空のSink sessionを返す。"""

        return _SinkSession(self.callback)
