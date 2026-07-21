"""v0.2の明示複数output Port契約を検証する。"""

import pytest

import chronowire as cw


def _values(result: cw.RunResult, index: int) -> list[object]:
    return [item.value for item in result.outputs[index].emissions]


def test_map_outputs_creates_fixed_flow_handles_and_port_descriptors() -> None:
    """KernelOutputsを固定数の通常Flow handleへ配送する。"""

    source = cw.Flow([1, 2])
    first, second = source.map_outputs(
        lambda value: cw.kernel_outputs(value, value * 10),
        output_count=2,
    )
    transformed = second.map(lambda value: value + 1)
    plan = cw.compile(
        [
            cw.output(first, collector=cw.Bounded(2)),
            cw.output(transformed, collector=cw.Bounded(2)),
        ]
    )

    result = plan.run()
    multi_node = next(item for item in plan.portable_ir.nodes if len(item.output_port_ids) == 2)
    multi_ports = [
        item for item in plan.portable_ir.ports if item.producer_node_id == multi_node.node_id
    ]

    assert _values(result, 0) == [1, 2]
    assert _values(result, 1) == [11, 21]
    assert [item.output_index for item in multi_ports] == [0, 1]
    assert tuple(item.port_id for item in multi_ports) == multi_node.output_port_ids


def test_unobserved_multi_output_does_not_retain_or_overflow() -> None:
    """requiredでないsibling Portは値を保持せず観測Portの実行を妨げない。"""

    source = cw.Flow(range(20))
    first, _unused = source.map_outputs(
        lambda value: cw.kernel_outputs(value, value * 100),
        output_count=2,
    )

    result = cw.compile([cw.output(first, collector=cw.Bounded(20))]).run()

    assert _values(result, 0) == list(range(20))
    assert result.completed


def test_multi_output_requires_kernel_outputs_not_ordinary_tuple() -> None:
    """v0.1の通常tuple一値契約を複数Portでも暗黙展開しない。"""

    source = cw.Flow([1])
    first, second = source.map_outputs(lambda value: (value, value), output_count=2)
    plan = cw.compile([first, second])

    with pytest.raises(cw.KernelExecutionError, match="requires KernelOutputs"):
        plan.run()
