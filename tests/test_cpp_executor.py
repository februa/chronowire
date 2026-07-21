"""compile済みPlanを自立運用する最小CppExecutorを検証する。"""

from collections.abc import Iterable

import pytest

import chronowire as cw
from chronowire.collector import Collector
from chronowire.cpp_executor import CppExecutionSession
from chronowire_reference import CythonCbfBackend, FixedCbfKernel


def _plan(
    *,
    collector: Collector[object] | None = None,
    source_values: Iterable[tuple[int | float, ...] | cw.Emission[tuple[int | float, ...]]]
    | None = None,
) -> cw.ExecutionPlan:
    values = (
        [(float(index + 1), float(index + 1)) for index in range(8)]
        if source_values is None
        else source_values
    )
    source = cw.Flow(cw.f64_vector_source(values, width=2))
    beams = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    resolved_collector = cw.Bounded(4) if collector is None else collector
    return cw.compile(
        [cw.output(beams, collector=resolved_collector)],
        backend=CythonCbfBackend(),
    )


def test_cpp_executor_matches_python_and_cython_cbf_trace() -> None:
    """値、interval、sequence、status、Diagnosticを三Executorで一致させる。"""

    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "CPP_DEGRADED", "safe fallback")
    source = [
        cw.Emission(
            (float(index + 1), float(index + 1)),
            cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1)),
            index,
            cw.EmissionStatus.DEGRADED if index == 0 else cw.EmissionStatus.OK,
            (diagnostic,) if index == 0 else (),
        )
        for index in range(8)
    ]
    plan = _plan(source_values=source)

    python = plan.run(executor=cw.PythonExecutor())
    cython = plan.run(executor=cw.CythonExecutor())
    cpp = plan.run(executor=cw.CppExecutor())

    assert cpp == python == cython
    assert cpp.outputs[0].emissions[0].diagnostics == (diagnostic,)


@pytest.mark.parametrize(
    ("collector", "sequences", "dropped"),
    [
        (cw.Latest(), (3,), 3),
        (cw.Bounded(2, cw.OverflowPolicy.DROP_OLDEST), (2, 3), 2),
        (cw.Bounded(2, cw.OverflowPolicy.DROP_NEWEST), (0, 1), 2),
    ],
)
def test_cpp_executor_applies_collector_policy_in_native_runtime(
    collector: Collector[object],
    sequences: tuple[int, ...],
    dropped: int,
) -> None:
    """Latest/Boundedの保持対象をPythonへ全件復元する前に選択する。"""

    plan = _plan(collector=collector)

    cpp = plan.run(executor="cpp")
    python = plan.run(executor="python")

    assert cpp == python
    assert tuple(item.sequence for item in cpp.outputs[0].emissions) == sequences
    assert cpp.outputs[0].dropped_count == dropped


def test_cpp_executor_no_collect_avoids_output_value_boundary() -> None:
    """NoCollectではCBFを実行しつつ値をPythonへcopyしない。"""

    plan = _plan(collector=cw.NoCollect())
    session = plan.create_session(executor=cw.CppExecutor())
    assert isinstance(session, CppExecutionSession)

    result = session.run()

    assert result == plan.run(executor=cw.PythonExecutor())
    assert result.outputs[0].emissions == ()
    assert result.outputs[0].received_count == 4
    assert session.last_metrics is not None
    assert session.last_metrics.output_boundary_bytes == 0
    assert session.last_metrics.owned_input_bytes > 0
    assert session.last_metrics.python_native_transitions == 2
    assert session.last_metrics.stage_python_dispatches == 0


def test_cpp_executor_session_can_run_again_without_state_leak() -> None:
    """同じC++ sessionの再実行でもcursor、collector、statusを持ち越さない。"""

    session = _plan().create_session(executor="cpp")

    first = session.run()
    second = session.run()

    assert first == second


def test_cpp_executor_reports_bounded_fail_without_partial_result() -> None:
    """Bounded FAIL overflowをPython fallbackや部分結果に変換しない。"""

    plan = _plan(collector=cw.Bounded(2))

    with pytest.raises(cw.BufferOverflowError, match="capacity 2"):
        plan.run(executor="cpp")


def test_cpp_executor_rejects_invalid_partition_and_metadata_explicitly() -> None:
    """未実装StreamItem契約を値や理由の欠落として実行しない。"""

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    invalid = [cw.Emission((1.0, 1.0), interval, 0, cw.EmissionStatus.INVALID)]
    metadata = [cw.Emission((1.0, 1.0), interval, 0, metadata={"source": "test"})]

    with pytest.raises(ValueError, match="batch_invalid_partition"):
        _plan(source_values=invalid).run(executor="cpp")
    with pytest.raises(ValueError, match="metadata_table"):
        _plan(source_values=metadata).run(executor="cpp")


def test_cpp_executor_rejects_continuous_session_without_fallback() -> None:
    """未実装PlanSessionをPythonExecutorへ暗黙委譲しない。"""

    with pytest.raises(cw.PlanSessionError, match="cpp_continuous_session"):
        _plan().create_plan_session(executor=cw.CppExecutor())
