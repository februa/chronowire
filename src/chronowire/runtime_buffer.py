"""run-localな共有PortBufferとconsumer cursorを定義する。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

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

    境界条件:
        itemは全consumerが通過するまで一度だけ保持する。consumerがない
        Portへpublishしたitemは保持しない。consumerごとの読み取り順序はFIFOである。

    Raises:
        KeyError: 未登録cursorを操作した場合。
        ValueError: cursor IDを重複登録した場合。
    """

    def __init__(self, buffer_id: int) -> None:
        self.buffer_id = buffer_id
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

    def publish(self, item: T) -> None:
        """itemを一度だけ末尾へ追加する。

        consumerがない場合は同期観測完了後の値を保持しない。
        """

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

    def __bool__(self) -> bool:
        return self._buffer.pending_count(self._cursor_id) > 0

    def __len__(self) -> int:
        return self._buffer.pending_count(self._cursor_id)

    def __getitem__(self, index: int) -> T:
        if index != 0:
            raise IndexError("CursorQueue supports only head access")
        item = self._buffer.peek(self._cursor_id)
        if item is None:
            raise IndexError("CursorQueue is empty")
        return item

    def popleft(self) -> T:
        """先頭itemを返して対応consumer cursorを一件進める。"""

        return self._buffer.pop(self._cursor_id)
