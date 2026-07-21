"""run-localな共有PortBufferとconsumer cursorを定義する。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
from typing import Generic, TypeVar

from .model import Emission, LogicalInterval
from .source import RealtimeOverflowPolicy

T = TypeVar("T")


@dataclass
class ConsumerCursor:
    """一つのconsumerが次に読むPortBuffer位置を保持する。

    cursorはrun-localであり、GraphやExecutionPlanへ可変状態を持ち込まない。
    利用者はpositionを直接変更せず、PortBufferの操作を通して進める。
    """

    cursor_id: int
    position: int


class PortBuffer(Generic[T]):
    """一つのPort出力をfan-out consumer間で共有する読み取り専用buffer。

    Args:
        buffer_id: PortablePlanIRと対応するrun-local buffer ID。
        max_items: 同時保持できるitem数。正の整数。

    境界条件:
        itemは全consumerが通過するまで一度だけ保持する。consumerがない
        Portへpublishしたitemは保持しない。consumerごとの読み取り順序はFIFOである。

    Raises:
        KeyError: 未登録cursorを操作した場合。
        ValueError: cursor IDを重複登録した場合。
    """

    def __init__(self, buffer_id: int, max_items: int = 1) -> None:
        if max_items <= 0:
            raise ValueError("PortBuffer max_items must be positive")
        self.buffer_id = buffer_id
        self.max_items = max_items
        self._items: deque[T] = deque()
        self._start_position = 0
        self._next_position = 0
        self._cursors: dict[int, ConsumerCursor] = {}
        self._high_watermark = 0

    def register_consumer(self, cursor_id: int) -> ConsumerCursor:
        """現在の末尾から読み始めるconsumer cursorを登録する。

        Args:
            cursor_id: ExecutionPlan内で一意なconsumer cursor ID。

        Returns:
            PortBufferが所有するrun-local cursor。

        Raises:
            ValueError: 同じcursor IDがすでに登録済みの場合。
        """

        if cursor_id in self._cursors:
            raise ValueError(f"buffer {self.buffer_id} already has cursor {cursor_id}")
        cursor = ConsumerCursor(cursor_id, self._next_position)
        self._cursors[cursor_id] = cursor
        return cursor

    def unregister_consumer(self, cursor_id: int) -> None:
        """停止したconsumerを解除し、不要になった共有prefixを解放する。

        Raises:
            KeyError: cursorが登録されていない場合。
        """

        self._cursor(cursor_id)
        del self._cursors[cursor_id]
        self._reclaim_consumed_prefix()

    def can_publish(self, count: int = 1) -> bool:
        """count件をdropなしで原子的に追加できる場合にTrueを返す。"""

        if count < 0:
            raise ValueError("publish count must not be negative")
        return not self._cursors or len(self._items) + count <= self.max_items

    def publish(self, item: T) -> None:
        """itemを一度だけ末尾へ追加する。

        consumerがない場合は同期観測完了後の値を保持しない。
        """

        if not self.can_publish():
            raise BufferError(
                f"buffer {self.buffer_id} capacity {self.max_items} would be exceeded"
            )
        self._next_position += 1
        if not self._cursors:
            self._start_position = self._next_position
            return
        self._items.append(item)
        self._high_watermark = max(self._high_watermark, len(self._items))

    def peek(self, cursor_id: int) -> T | None:
        """cursor位置のitemを進めずに返す。未到着ならNoneを返す。"""

        cursor = self._cursor(cursor_id)
        if cursor.position >= self._next_position:
            return None
        offset = cursor.position - self._start_position
        if offset < 0 or offset >= len(self._items):
            raise RuntimeError(
                f"buffer {self.buffer_id} lost cursor {cursor_id} position {cursor.position}"
            )
        return self._items[offset]

    def pop(self, cursor_id: int) -> T:
        """cursor位置のitemを返して一件進める。

        Raises:
            IndexError: cursor位置にitemがまだ到着していない場合。
        """

        item = self.peek(cursor_id)
        if item is None:
            raise IndexError(f"buffer {self.buffer_id} cursor {cursor_id} has no pending item")
        self._cursors[cursor_id].position += 1
        self._reclaim_consumed_prefix()
        return item

    def pending_count(self, cursor_id: int) -> int:
        """指定consumerが未処理のitem数を返す。"""

        cursor = self._cursor(cursor_id)
        return self._next_position - cursor.position

    @property
    def retained_count(self) -> int:
        """いずれかのconsumerのために現在保持しているitem数を返す。"""

        return len(self._items)

    @property
    def high_watermark(self) -> int:
        """このrunで同時保持した最大item数を返す。"""

        return self._high_watermark

    @property
    def consumer_count(self) -> int:
        """登録済みconsumer cursor数を返す。"""

        return len(self._cursors)

    def _cursor(self, cursor_id: int) -> ConsumerCursor:
        try:
            return self._cursors[cursor_id]
        except KeyError as error:
            raise KeyError(f"buffer {self.buffer_id} has no cursor {cursor_id}") from error

    def _reclaim_consumed_prefix(self) -> None:
        minimum_position = min(
            (cursor.position for cursor in self._cursors.values()),
            default=self._next_position,
        )
        reclaim_count = minimum_position - self._start_position
        for _ in range(reclaim_count):
            self._items.popleft()
        self._start_position = minimum_position


class CursorQueue(Generic[T]):
    """既存SchedulerのFIFO操作をPortBuffer cursorへ写像する内部adapter。"""

    def __init__(self, buffer: PortBuffer[T], cursor_id: int) -> None:
        self._buffer = buffer
        self._cursor_id = cursor_id
        self._closed = False

    def __bool__(self) -> bool:
        return not self._closed and self._buffer.pending_count(self._cursor_id) > 0

    def __len__(self) -> int:
        if self._closed:
            return 0
        return self._buffer.pending_count(self._cursor_id)

    def __getitem__(self, index: int) -> T:
        if self._closed:
            raise IndexError("CursorQueue is closed")
        if index != 0:
            raise IndexError("CursorQueue supports only head access")
        item = self._buffer.peek(self._cursor_id)
        if item is None:
            raise IndexError("CursorQueue is empty")
        return item

    def popleft(self) -> T:
        """先頭itemを返して対応consumer cursorを一件進める。"""

        return self._buffer.pop(self._cursor_id)

    def close(self) -> None:
        """consumerを停止し、以後このqueueを空として扱う。"""

        if self._closed:
            return
        self._buffer.unregister_consumer(self._cursor_id)
        self._closed = True


class FrameHistoryBuffer(Generic[T]):
    """一つのFRAME Nodeが未確定入力履歴を保持するrun-local buffer。

    Args:
        buffer_id: PortablePlanIRの`FRAME_HISTORY` buffer ID。
        max_items: frame sizeと一致する正の保持上限。

    境界条件:
        item参照はcopyせず、frame確定またはhop前進時だけ明示的に解放する。

    Raises:
        BufferError: compile済みcapacityを越えて履歴を追加した場合。
    """

    def __init__(self, buffer_id: int, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError("FrameHistoryBuffer max_items must be positive")
        self.buffer_id = buffer_id
        self.max_items = max_items
        self._items: list[T] = []
        self._high_watermark = 0

    def append(self, item: T) -> None:
        """一件を履歴末尾へ追加する。"""

        if len(self._items) >= self.max_items:
            raise BufferError(
                f"frame history buffer {self.buffer_id} capacity {self.max_items} would be exceeded"
            )
        self._items.append(item)
        self._high_watermark = max(self._high_watermark, len(self._items))

    def snapshot(self, count: int | None = None) -> tuple[T, ...]:
        """現在の履歴prefixを不変tupleとして返す。"""

        return tuple(self._items if count is None else self._items[:count])

    def discard_prefix(self, count: int) -> None:
        """hopで不要になったprefixを解放する。"""

        if count < 0:
            raise ValueError("discard count must not be negative")
        del self._items[:count]

    def clear(self) -> None:
        """欠落、EOF flush、またはrun終了境界で全履歴を解放する。"""

        self._items.clear()

    @property
    def first(self) -> T | None:
        """最古itemを返し、空ならNoneを返す。"""

        return self._items[0] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def high_watermark(self) -> int:
        """このrunで保持した最大item数を返す。"""

        return self._high_watermark


class LatestStateBuffer(Generic[T]):
    """一つのLATEST入力について確定済み最新値一件を保持するbuffer。

    Args:
        buffer_id: PortablePlanIRの`LATEST_STATE` buffer ID。

    境界条件:
        新しい確定値は古い値を置換し、future pendingは共有PortBuffer側に残す。
    """

    def __init__(self, buffer_id: int) -> None:
        self.buffer_id = buffer_id
        self._value: T | None = None
        self._has_value = False

    def replace(self, item: T) -> None:
        """確定済み最新値を一件で置換する。"""

        self._value = item
        self._has_value = True

    @property
    def has_value(self) -> bool:
        """一件以上の確定値を保持している場合にTrueを返す。"""

        return self._has_value

    def get(self) -> T:
        """確定済み最新値を返す。

        Raises:
            LookupError: まだ値が確定していない場合。
        """

        if not self._has_value:
            raise LookupError(f"latest state buffer {self.buffer_id} has no value")
        value = self._value
        assert value is not None
        return value


@dataclass(frozen=True)
class GapMarker:
    """Realtime ingressで失われた連続intervalを表す内部control record。"""

    source_node_id: int
    source_port_id: int
    interval: LogicalInterval
    dropped_count: int
    total_dropped_count: int
    capacity: int
    overflow_policy: RealtimeOverflowPolicy


class RealtimeIngressBuffer(Generic[T]):
    """外部callbackとSchedulerを分離するthread-safe bounded ingress。

    Args:
        buffer_id: PortablePlanIRの`REALTIME_INGRESS` buffer ID。
        source_node_id: Source Node ID。
        source_port_id: Source output Port ID。
        max_items: 通常Emissionの保持上限。
        overflow_policy: 満杯時の棄却規則。blockingは提供しない。

    境界条件:
        GapMarkerは通常item数へ数えず、隣接dropをcoalesceして失わない。
    """

    def __init__(
        self,
        buffer_id: int,
        source_node_id: int,
        source_port_id: int,
        max_items: int,
        overflow_policy: RealtimeOverflowPolicy,
    ) -> None:
        if max_items <= 0:
            raise ValueError("RealtimeIngressBuffer max_items must be positive")
        self.buffer_id = buffer_id
        self.source_node_id = source_node_id
        self.source_port_id = source_port_id
        self.max_items = max_items
        self.overflow_policy = overflow_policy
        self._records: deque[Emission[T] | GapMarker] = deque()
        self._item_count = 0
        self._high_watermark = 0
        self._total_dropped = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._condition = Condition()

    def publish(self, emission: Emission[T]) -> None:
        """一件を非blockingで受理し、満杯ならpolicyに従って棄却する。"""

        if not isinstance(emission, Emission):
            raise TypeError("realtime receiver accepts only Emission values")
        with self._condition:
            if self._closed:
                return
            if self._item_count < self.max_items:
                self._records.append(emission)
                self._item_count += 1
            elif self.overflow_policy is RealtimeOverflowPolicy.DROP_NEWEST:
                self._record_gap(len(self._records), emission)
            else:
                oldest_index = next(
                    index
                    for index, record in enumerate(self._records)
                    if isinstance(record, Emission)
                )
                dropped = self._records[oldest_index]
                if not isinstance(dropped, Emission):
                    raise RuntimeError("realtime ingress lost its oldest Emission")
                del self._records[oldest_index]
                self._item_count -= 1
                self._record_gap(oldest_index, dropped)
                self._records.append(emission)
                self._item_count += 1
            self._high_watermark = max(self._high_watermark, self._item_count)
            self._condition.notify()

    def close(self) -> None:
        """受付を正常終了し、残るrecordのdrain後にEOFとする。"""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        """受付を失敗終了し、drain後にSchedulerへ原因を送出する。"""

        with self._condition:
            self._failure = error
            self._closed = True
            self._condition.notify_all()

    def take(self) -> Emission[T] | GapMarker | None:
        """次recordを待って返す。正常close後のdrain完了時はNoneを返す。"""

        with self._condition:
            while not self._records and not self._closed:
                self._condition.wait()
            if self._records:
                record = self._records.popleft()
                if isinstance(record, Emission):
                    self._item_count -= 1
                return record
            if self._failure is not None:
                raise RuntimeError(
                    f"realtime source node {self.source_node_id} port {self.source_port_id} failed"
                ) from self._failure
            return None

    def discard(self) -> int:
        """cancel時に未処理recordを破棄し、通常Emission件数を返す。"""

        with self._condition:
            discarded = self._item_count
            self._records.clear()
            self._item_count = 0
            self._closed = True
            self._condition.notify_all()
            return discarded

    @property
    def is_closed(self) -> bool:
        """外部受付が終了している場合にTrueを返す。"""

        with self._condition:
            return self._closed

    @property
    def pending_count(self) -> int:
        """drain待ちの通常Emission件数を返す。"""

        with self._condition:
            return self._item_count

    @property
    def high_watermark(self) -> int:
        """session中に同時保持した通常Emission最大件数を返す。"""

        with self._condition:
            return self._high_watermark

    @property
    def total_dropped_count(self) -> int:
        """このrunで棄却したEmission総数を返す。"""

        with self._condition:
            return self._total_dropped

    def _record_gap(self, index: int, emission: Emission[T]) -> None:
        self._total_dropped += 1
        marker = GapMarker(
            self.source_node_id,
            self.source_port_id,
            emission.interval,
            1,
            self._total_dropped,
            self.max_items,
            self.overflow_policy,
        )
        previous = self._records[index - 1] if index > 0 else None
        if isinstance(previous, GapMarker) and previous.interval.end == marker.interval.start:
            self._records[index - 1] = GapMarker(
                previous.source_node_id,
                previous.source_port_id,
                LogicalInterval(previous.interval.start, marker.interval.end),
                previous.dropped_count + 1,
                marker.total_dropped_count,
                previous.capacity,
                previous.overflow_policy,
            )
            return
        self._records.insert(index, marker)
