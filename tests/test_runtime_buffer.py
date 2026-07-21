"""共有PortBufferとconsumer cursorの寿命契約を検証する。"""

import pytest

import chronowire as cw
from chronowire.runtime_buffer import (
    CursorQueue,
    FrameHistoryBuffer,
    GapMarker,
    LatestStateBuffer,
    PortBuffer,
    RealtimeIngressBuffer,
)


def test_fan_out_keeps_one_shared_item_until_all_consumers_advance() -> None:
    """fan-out itemを一度だけ保持し、最後のconsumer通過後に解放する。"""

    buffer = PortBuffer[object](buffer_id=7)
    first = buffer.register_consumer(10)
    second = buffer.register_consumer(11)
    item = object()

    buffer.publish(item)

    assert buffer.retained_count == 1
    assert buffer.peek(first.cursor_id) is item
    assert buffer.peek(second.cursor_id) is item
    assert buffer.pop(first.cursor_id) is item
    assert buffer.retained_count == 1
    assert buffer.pop(second.cursor_id) is item
    assert buffer.retained_count == 0
    assert buffer.high_watermark == 1


def test_cursor_queue_preserves_independent_fifo_positions() -> None:
    """consumerごとのFIFO位置が互いに影響せず、値本体だけを共有する。"""

    buffer = PortBuffer[int](buffer_id=3, max_items=2)
    buffer.register_consumer(20)
    buffer.register_consumer(21)
    first = CursorQueue(buffer, 20)
    second = CursorQueue(buffer, 21)
    buffer.publish(1)
    buffer.publish(2)

    assert len(first) == 2
    assert len(second) == 2
    assert first.popleft() == 1
    assert len(first) == 1
    assert len(second) == 2
    assert second.popleft() == 1
    assert second.popleft() == 2
    assert buffer.retained_count == 1
    assert first.popleft() == 2
    assert buffer.retained_count == 0


def test_port_buffer_rejects_unknown_or_duplicate_cursor() -> None:
    """cursor登録違反を曖昧な空bufferとして扱わない。"""

    buffer = PortBuffer[int](buffer_id=4)
    buffer.register_consumer(30)

    with pytest.raises(ValueError, match="already has cursor"):
        buffer.register_consumer(30)
    with pytest.raises(KeyError, match="has no cursor"):
        buffer.peek(99)


def test_port_without_consumers_does_not_retain_items() -> None:
    """観測配送後に下流consumerがないPortは値を保持しない。"""

    buffer = PortBuffer[int](buffer_id=5)
    buffer.publish(1)

    assert buffer.retained_count == 0
    assert buffer.high_watermark == 0


def test_capacity_applies_to_shared_retention_without_partial_publish() -> None:
    """遅いconsumerがいる場合は保持上限を越えるpublishを拒否する。"""

    buffer = PortBuffer[int](buffer_id=6, max_items=2)
    buffer.register_consumer(40)
    buffer.publish(1)
    buffer.publish(2)

    assert not buffer.can_publish()
    with pytest.raises(BufferError, match="capacity 2"):
        buffer.publish(3)
    assert buffer.retained_count == 2


def test_closing_stalled_cursor_reclaims_shared_prefix() -> None:
    """停止Nodeのcursor解除後は他consumerが通過済みのprefixを解放する。"""

    buffer = PortBuffer[int](buffer_id=8, max_items=2)
    buffer.register_consumer(50)
    buffer.register_consumer(51)
    fast = CursorQueue(buffer, 50)
    stalled = CursorQueue(buffer, 51)
    buffer.publish(1)
    assert fast.popleft() == 1

    stalled.close()

    assert buffer.retained_count == 0
    assert not stalled


def test_frame_history_has_explicit_capacity_and_hop_reclaim() -> None:
    """FRAME履歴はPlan容量を越えず、hop prefixだけを解放する。"""

    history = FrameHistoryBuffer[int](buffer_id=20, max_items=3)
    history.append(1)
    history.append(2)
    history.append(3)

    assert history.snapshot() == (1, 2, 3)
    assert history.high_watermark == 3
    with pytest.raises(BufferError, match="capacity 3"):
        history.append(4)

    history.discard_prefix(2)
    assert history.snapshot() == (3,)


def test_latest_state_replaces_old_value_without_queue_growth() -> None:
    """LATEST_STATEは確定値一件だけを置換保持する。"""

    state = LatestStateBuffer[int](buffer_id=21)
    assert not state.has_value
    with pytest.raises(LookupError, match="has no value"):
        state.get()

    state.replace(1)
    state.replace(2)

    assert state.has_value
    assert state.get() == 2


def _emission(index: int) -> cw.Emission[int]:
    return cw.Emission(
        index,
        cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1)),
        index,
    )


def test_realtime_ingress_drop_oldest_coalesces_gap_before_retained_items() -> None:
    """DROP_OLDESTが連続欠落を失わず、保持itemより先に配送する。"""

    ingress = RealtimeIngressBuffer[int](
        30,
        0,
        0,
        2,
        cw.RealtimeOverflowPolicy.DROP_OLDEST,
    )
    for index in range(4):
        ingress.publish(_emission(index))
    ingress.close()

    gap = ingress.take()
    assert isinstance(gap, GapMarker)
    assert gap.interval == cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(2))
    assert gap.dropped_count == 2
    assert gap.total_dropped_count == 2
    assert ingress.take() == _emission(2)
    assert ingress.take() == _emission(3)
    assert ingress.take() is None


def test_realtime_ingress_drop_newest_preserves_buffered_items() -> None:
    """DROP_NEWESTは既存itemを保持し、後続欠落を末尾へ記録する。"""

    ingress = RealtimeIngressBuffer[int](
        31,
        0,
        0,
        2,
        cw.RealtimeOverflowPolicy.DROP_NEWEST,
    )
    for index in range(4):
        ingress.publish(_emission(index))
    ingress.close()

    assert ingress.take() == _emission(0)
    assert ingress.take() == _emission(1)
    gap = ingress.take()
    assert isinstance(gap, GapMarker)
    assert gap.interval == cw.LogicalInterval(cw.LogicalTime(2), cw.LogicalTime(4))
    assert gap.dropped_count == 2
    assert ingress.take() is None
