"""compile済みPlanを自立運用する最小CppExecutorを検証する。"""

from collections.abc import Iterable
from fractions import Fraction

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


def test_cpp_executor_preserves_invalid_partition_and_metadata() -> None:
    """INVALID pass-throughと途中Port metadataをPython基準意味論と一致させる。"""

    first = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    second = cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(2))
    values = [
        cw.Emission(
            (1.0, 1.0),
            first,
            0,
            cw.EmissionStatus.INVALID,
            metadata={"source": "test"},
        ),
        cw.Emission((2.0, 2.0), second, 1),
    ]
    source = cw.Flow(cw.f64_vector_source(values, width=2))
    rated = source.rate(1)
    beams = rated.frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    plan = cw.compile(
        [
            cw.output(rated, collector=cw.Bounded(2)),
            cw.output(beams, collector=cw.Bounded(1)),
        ],
        backend=CythonCbfBackend(),
    )

    cpp = plan.run(executor="cpp")

    assert cpp == plan.run(executor="python")
    assert cpp.outputs[0].emissions[0].metadata == {"source": "test"}
    assert cpp.outputs[1].emissions[0].status is cw.EmissionStatus.INVALID
    assert cpp.outputs[1].emissions[0].diagnostics[-1].code == "INVALID_INPUT_PROPAGATED"


def test_cpp_executor_plan_session_advances_monotonically_and_drains() -> None:
    """有限native Planを論理時間境界ごとにC++で累積実行する。"""

    plan = _plan()
    session = plan.create_plan_session(executor=cw.CppExecutor())

    assert session.state is cw.PlanSessionState.CREATED
    session.start()
    assert session.run_until(Fraction(1, 1)).outputs[0].emissions == ()
    first_frame = session.run_until(2)
    assert first_frame.outputs[0].emissions == plan.run(duration=2).outputs[0].emissions
    with pytest.raises(cw.PlanSessionError, match="strictly increasing"):
        session.run_until(2)
    closed = session.close()

    assert closed.outputs == plan.run().outputs
    assert closed.completed
    assert session.state is cw.PlanSessionState.CLOSED


def test_cpp_executor_plan_session_cancel_preserves_last_snapshot() -> None:
    """cancelはdrainせず直前snapshotと明示Diagnosticを保持する。"""

    session = _plan().create_plan_session(executor="cpp")
    session.start()
    observed = session.run_until(Fraction(2, 1))

    cancelled = session.cancel()

    assert cancelled.outputs == observed.outputs
    assert not cancelled.completed
    assert cancelled.diagnostics[-1].code == "SESSION_CANCELLED"
    assert session.state is cw.PlanSessionState.CANCELLED
    with pytest.raises(cw.PlanSessionError, match="state=running"):
        session.flush()


def test_cpp_executor_runs_kernel_chain_and_fanout_ancestor_once() -> None:
    """複数native Stageとfan-outで共通RATE/FRAMEを二重評価しない。"""

    source = cw.Flow(
        cw.f64_vector_source(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
            width=2,
        )
    )
    frames = source.rate(1).frame(2)
    first = frames.map(FixedCbfKernel(((0.5, 0.5),))).map(cw.identity_f64())
    second = frames.map(FixedCbfKernel(((1.0, 0.0),)))
    plan = cw.compile(
        [
            cw.output(first, collector=cw.Bounded(2)),
            cw.output(second, collector=cw.Bounded(2)),
        ],
        backend=CythonCbfBackend(),
    )
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert result == plan.run(executor="python")
    assert isinstance(session, CppExecutionSession)
    assert session.last_metrics is not None
    assert session.last_metrics.executed_node_count == len(plan.portable_ir.nodes)
    assert len(plan.portable_ir.kernel_abis) == 3


def test_cpp_executor_long_streaming_cbf_matches_python() -> None:
    """長いCBF入力でも値、時刻、status、保持境界をPythonと一致させる。"""

    sample_count = 8192
    source = cw.Flow(
        cw.f64_vector_source(
            [
                (float(index % 17), float((index * 3) % 19), float((index * 5) % 23), 1.0)
                for index in range(sample_count)
            ],
            width=4,
        )
    )
    beams = (
        source.rate(1)
        .frame(64, hop=32)
        .map(FixedCbfKernel(((0.25, 0.25, 0.25, 0.25), (1.0, 0.0, 0.0, 0.0))))
    )
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(8, cw.OverflowPolicy.DROP_OLDEST))],
        backend=CythonCbfBackend(),
    )

    cpp = plan.run(executor="cpp")

    assert cpp == plan.run(executor="python")
    assert cpp.outputs[0].received_count == 255
    assert len(cpp.outputs[0].emissions) == 8


def test_cpp_executor_resets_rate_and_frame_at_input_overrun() -> None:
    """INPUT_OVERRUN境界を跨ぐframeを作らずPythonと同じ二frameを生成する。"""

    overrun = cw.Diagnostic(cw.Severity.WARNING, "INPUT_OVERRUN", "dropped samples")
    values = [
        cw.Emission(
            (float(tick), float(tick)),
            cw.LogicalInterval(cw.LogicalTime(tick), cw.LogicalTime(tick + 1)),
            index,
            diagnostics=(overrun,) if tick == 4 else (),
        )
        for index, tick in enumerate((0, 1, 4, 5))
    ]
    source = cw.Flow(cw.f64_vector_source(values, width=2))
    beams = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(2))],
        backend=CythonCbfBackend(),
    )

    result = plan.run(executor="cpp")

    assert result == plan.run(executor="python")
    assert tuple(item.interval.start.ticks for item in result.outputs[0].emissions) == (0, 4)


class _RecordingSession:
    def __init__(self, events: list[cw.Emission[object]]) -> None:
        self.events = events
        self.finalized = False

    def initialize(self, context: cw.PlanContext) -> None:
        assert context.required_node_count > 0

    def on_output(self, event: cw.OutputEvent) -> None:
        self.events.append(event.emission)

    def on_diagnostic(self, diagnostic: cw.Diagnostic) -> None:
        return

    def finalize(self, context: cw.PlanContext) -> None:
        self.finalized = True


class _RecordingExtension:
    abi_version = "chronowire.extension.v1"

    def __init__(self) -> None:
        self.sessions: list[_RecordingSession] = []

    def create_session(self) -> _RecordingSession:
        session = _RecordingSession([])
        self.sessions.append(session)
        return session


def test_cpp_executor_delivers_extension_at_python_stage_boundary() -> None:
    """C++観測Portからtrigger済みEmissionだけをPython Extensionへ配送する。"""

    source = cw.Flow(
        cw.f64_vector_source(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
            width=2,
        )
    )
    beams = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    observation = cw.observe(beams, extension_id="record", trigger=cw.Every(2))
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(2))],
        extensions=[observation],
        backend=CythonCbfBackend(),
    )
    extension = _RecordingExtension()

    result = plan.create_session(executor="cpp", extension_bindings={"record": extension}).run()

    assert extension.sessions[0].events == [result.outputs[0].emissions[0]]
    assert extension.sessions[0].finalized
