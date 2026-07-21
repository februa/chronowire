"""共有PortBufferとconsumer cursorの寿命契約を検証する。"""

import pytest

from chronowire.runtime_buffer import CursorQueue, PortBuffer


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

    buffer = PortBuffer[int](buffer_id=3)
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
