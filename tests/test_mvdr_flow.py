"""一定論理間隔で更新するMVDR FlowのPython/C++同値性を検証する。"""

from __future__ import annotations

from fractions import Fraction

import pytest

import chronowire as cw
from chronowire_reference import MvdrNativeBackend, build_mvdr_flow

_SAMPLES = (
    (1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 0.0),
)


def _plan() -> cw.Plan:
    """重み更新とbeam出力をともに観測するnative MVDR Planを返す。"""

    flow = build_mvdr_flow(_SAMPLES, frame_size=2, update_period=4)
    return cw.compile(
        [
            cw.output(flow.beam, collector=cw.Bounded(4)),
            cw.output(flow.weight_updates, collector=cw.Bounded(2)),
        ],
        backend=MvdrNativeBackend(),
    )


def _trace(result: cw.RunResult) -> tuple[tuple[object, ...], ...]:
    """Executor非依存に比較する値・時間・status traceを返す。"""

    return tuple(
        tuple(
            (
                emission.value,
                emission.interval.start.as_fraction(),
                emission.interval.end.as_fraction(),
                emission.sequence,
                emission.status,
                tuple(item.code for item in emission.diagnostics),
            )
            for emission in output.emissions
        )
        for output in result.outputs
    )


def test_mvdr_flow_compiles_periodic_update_and_latest_weight_edges() -> None:
    """更新周期とlatest weight適用をPortablePlanIRへ明示する。"""

    plan = _plan()
    ir = plan.portable_ir

    assert ir.schema_version == "0.4"
    sample = next(item for item in ir.nodes if item.rate_policy == "sample")
    assert sample.rate_period is not None
    assert Fraction(sample.rate_period.numerator, sample.rate_period.denominator) == 4
    apply = next(
        item
        for item in ir.operations
        if item.operation_id == "chronowire.reference.apply_weights_f64.v1"
    )
    apply_edges = [item for item in ir.edges if item.target_node_id == apply.node_id]
    assert [item.semantics for item in apply_edges] == ["synchronous", "latest"]
    assert [item.name for item in apply.inputs] == ["signal", "weights"]
    assert all(item.native_compatible for item in ir.implementations)


def test_mvdr_weight_updates_are_held_between_logical_boundaries() -> None:
    """重みを0、4秒だけ更新し、中間frameでは直前重みを保持する。"""

    result = _plan().run(executor=cw.PythonExecutor())
    beams = result.outputs[0].emissions
    weights = result.outputs[1].emissions

    assert [item.interval.start.as_fraction() for item in weights] == [Fraction(0), Fraction(4)]
    assert weights[0].value == pytest.approx((1.0 / 6.0, 5.0 / 6.0))
    assert weights[1].value == pytest.approx((11.0 / 18.0, 7.0 / 18.0))
    assert weights[0].status is cw.EmissionStatus.DEGRADED
    assert weights[0].diagnostics[0].code == "INSUFFICIENT_INTEGRATION"
    assert weights[1].status is cw.EmissionStatus.OK
    assert [item.interval.start.as_fraction() for item in beams] == [
        Fraction(0),
        Fraction(2),
        Fraction(4),
        Fraction(6),
    ]
    assert beams[0].value == pytest.approx((1.0 / 6.0, 1.0 / 6.0))
    assert beams[1].value == pytest.approx((5.0 / 6.0, 5.0 / 6.0))
    assert beams[2].value == pytest.approx((7.0 / 18.0, 7.0 / 18.0))
    assert beams[3].value == pytest.approx((11.0 / 18.0, 11.0 / 18.0))
    assert [item.status for item in beams] == [
        cw.EmissionStatus.DEGRADED,
        cw.EmissionStatus.DEGRADED,
        cw.EmissionStatus.OK,
        cw.EmissionStatus.OK,
    ]


def test_mvdr_cpp_executor_matches_python_and_resets_between_runs() -> None:
    """Plan運用全体をC++へ移してもtraceとrun-local更新状態を保つ。"""

    plan = _plan()
    python = _trace(plan.run(executor=cw.PythonExecutor()))
    first_cpp = _trace(plan.run(executor=cw.CppExecutor()))
    second_cpp = _trace(plan.run(executor=cw.CppExecutor()))

    assert first_cpp == python
    assert second_cpp == python


def test_sample_every_rejects_fractional_frame_selection_grid() -> None:
    """完成frameの端数を生む選択周期をcompile違反にする。"""

    sampled = cw.Flow(cw.f64_source([1.0, 2.0, 3.0, 4.0])).frame(2).sample_every(3)

    with pytest.raises(cw.CompileError, match="contract=stable_sample_boundary"):
        cw.compile([sampled])


def test_sample_every_expands_shared_fanout_capacity_for_update_lookahead() -> None:
    """低頻度更新branchが主branchを塞がないcapacityをcompileする。"""

    flow = build_mvdr_flow(_SAMPLES, frame_size=2, update_period=4)
    plan = cw.compile([flow.beam], backend=MvdrNativeBackend())
    frame_buffer = next(
        item
        for item in plan.portable_ir.buffers
        if item.kind == "port_shared" and item.producer_port_id == flow.frames.port_id
    )

    assert frame_buffer.max_items == 2
    assert any("shared_merge_demand" in item for item in frame_buffer.capacity_reasons)
