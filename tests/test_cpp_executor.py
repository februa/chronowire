"""compile済みPlanを自立運用する最小CppExecutorを検証する。"""

from collections.abc import Iterable, Mapping
from fractions import Fraction

import pytest

import chronowire as cw
from chronowire._cpp_executor import CppCooperativeStageSession
from chronowire.collector import Collector
from chronowire.cpp_executor import (
    CppMixedSession,
    CppPythonPrefixSession,
    CppPythonStageSession,
    CppSession,
)
from chronowire_reference import CythonCbfBackend, FixedCbfKernel


def _plan(
    *,
    collector: Collector[object] | None = None,
    source_values: Iterable[tuple[int | float, ...] | cw.Emission[tuple[int | float, ...]]]
    | None = None,
) -> cw.Plan:
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


def test_cpp_executor_runs_all_python_plan_as_one_cooperative_island() -> None:
    """全Python処理を関数ごとでなく最大Stage単位でyieldする。"""

    calls: list[tuple[str, int]] = []

    def double(value: int) -> int:
        calls.append(("double", value))
        return value * 2

    def increment(value: int) -> int:
        calls.append(("increment", value))
        return value + 1

    mapped = cw.Flow([1, 2, 3]).map(double).map(increment)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(3))])
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert isinstance(session, CppPythonStageSession)
    assert result == plan.run(executor="python")
    assert [item.value for item in result.outputs[0].emissions] == [3, 5, 7]
    assert len(plan.portable_ir.stages) == 1
    assert plan.portable_ir.stages[0].node_ids == (0, 1, 2)
    assert session.last_metrics is not None
    assert session.last_metrics.execution_classification == "python_stage_dominated"
    assert session.last_metrics.stage_python_dispatches == 1
    assert session.last_metrics.gil_acquisitions == 1
    assert not session.last_metrics.python_free_hot_path


def test_cpp_python_island_preserves_fan_out_status_and_diagnostic() -> None:
    """Python islandの共通祖先を1回だけ実行し劣化理由を保持する。"""

    calls = 0
    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "PYTHON_DEGRADED", "test fallback")

    def shared(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    source = cw.Flow(
        [
            cw.Emission(
                2,
                cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
                0,
                cw.EmissionStatus.DEGRADED,
                (diagnostic,),
            )
        ]
    )
    common = source.map(shared)
    left = common.map(lambda value: value + 1)
    right = common.map(lambda value: value - 1)
    plan = cw.compile(
        [
            cw.output(left, collector=cw.Latest()),
            cw.output(right, collector=cw.Latest()),
        ]
    )

    result = plan.run(executor="cpp")

    assert calls == 1
    assert result == plan.run(executor="python")
    assert result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED
    assert result.outputs[0].emissions[0].diagnostics == (diagnostic,)


def test_cpp_python_island_preserves_zero_one_many_emissions() -> None:
    """Python Stageの0/1/複数Emission契約をCppExecutor選択時も保つ。"""

    def expand(value: int) -> object:
        if value == 0:
            return cw.skip()
        if value == 1:
            return [10, 11]
        return cw.emit_many([20, 21])

    flow = cw.Flow([0, 1, 2]).map(expand, max_items=2)
    plan = cw.compile([cw.output(flow, collector=cw.Bounded(3))])

    result = plan.run(executor="cpp")

    assert result == plan.run(executor="python")
    assert [item.value for item in result.outputs[0].emissions] == [[10, 11], 20, 21]


def test_cpp_mixed_native_prefix_dispatches_one_python_island_batch() -> None:
    """native CBF出力をfan-out付き末尾Python islandへ一回だけ渡す。"""

    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "MIXED_DEGRADED", "safe fallback")
    source_values = [
        cw.Emission(
            (float(index + 1), float(index + 1)),
            cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1)),
            index,
            cw.EmissionStatus.DEGRADED if index == 0 else cw.EmissionStatus.OK,
            (diagnostic,) if index == 0 else (),
        )
        for index in range(4)
    ]
    source = cw.Flow(cw.f64_vector_source(source_values, width=2))
    native = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    calls = {"left": 0, "right": 0}

    def left(value: object) -> object:
        calls["left"] += 1
        return value

    def right(value: object) -> object:
        calls["right"] += 1
        return value

    plan = cw.compile(
        [
            cw.output(native.map(left), collector=cw.Bounded(2)),
            cw.output(native.map(right), collector=cw.Bounded(2)),
        ],
        backend=CythonCbfBackend(),
    )
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert isinstance(session, CppMixedSession)
    assert calls == {"left": 2, "right": 2}
    assert result == plan.run(executor="python")
    assert result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED
    assert result.outputs[0].emissions[0].diagnostics == (diagnostic,)
    assert session.last_metrics is not None
    assert session.last_metrics.execution_classification == "hybrid"
    assert session.last_metrics.stage_python_dispatches == 1
    assert session.last_metrics.gil_acquisitions == 1
    assert session.last_metrics.stage_boundary_batches == 1
    assert session.last_metrics.copied_batches == 1
    assert not session.last_metrics.python_free_hot_path


def test_cpp_mixed_python_stage_preserves_zero_and_multiple_emissions() -> None:
    """mixed境界後もSkipとEmitManyを暗黙のlist展開なしで保持する。"""

    source = cw.Flow(
        cw.f64_vector_source(
            [(float(index + 1), float(index + 1)) for index in range(4)],
            width=2,
        )
    )
    native = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    calls = 0

    def expand(value: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return cw.skip()
        return cw.emit_many((value, value))

    expanded = native.map(expand, max_items=2)
    plan = cw.compile(
        [cw.output(expanded, collector=cw.Bounded(2))],
        backend=CythonCbfBackend(),
    )

    result = plan.run(executor="cpp")

    assert len(result.outputs[0].emissions) == 2
    calls = 0
    assert result == plan.run(executor="python")


def test_cpp_mixed_python_failure_does_not_poison_next_plan_run() -> None:
    """mixed Python Stage例外後も新しいrun-local sessionで同じPlanを再実行する。"""

    source = cw.Flow(cw.f64_vector_source([(1.0, 1.0), (2.0, 2.0)], width=2))
    native = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
    should_fail = True

    def fail_once(value: object) -> object:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("intentional mixed failure")
        return value

    plan = cw.compile(
        [cw.output(native.map(fail_once), collector=cw.Latest())],
        backend=CythonCbfBackend(),
    )

    with pytest.raises(cw.KernelExecutionError, match="intentional mixed failure"):
        plan.run(executor="cpp")

    assert plan.run(executor="cpp").outputs[0].emissions


def test_cpp_python_prefix_resumes_native_cbf_suffix() -> None:
    """Python前処理とRATE/FRAMEのbatchをnative CBF suffixへresumeする。"""

    @cw.operation(
        operation_id="test.cpp_prefix_scale.v1",
        inputs={
            "input": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            )
        },
        output="same",
    )
    def scale(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        value = inputs["input"]
        assert isinstance(value, tuple)
        return tuple(float(item) * 2.0 for item in value)

    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "PREFIX_DEGRADED", "safe fallback")
    values = [
        cw.Emission(
            (float(index + 1), float(index + 1)),
            cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1)),
            index,
            cw.EmissionStatus.DEGRADED if index == 0 else cw.EmissionStatus.OK,
            (diagnostic,) if index == 0 else (),
        )
        for index in range(4)
    ]
    source = cw.Flow(cw.f64_vector_source(values, width=2))
    frames = source.map(scale).rate(1).frame(2)
    beams = frames.map(FixedCbfKernel(((0.5, 0.5),)))
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(2))],
        backend=CythonCbfBackend(),
        implementations={scale.operation_id: "python"},
    )
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert isinstance(session, CppPythonPrefixSession)
    assert result == plan.run(executor="python")
    assert result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED
    assert result.outputs[0].emissions[0].diagnostics == (diagnostic,)
    assert session.last_metrics is not None
    assert session.last_metrics.execution_classification == "hybrid"
    assert session.last_metrics.stage_python_dispatches == 1
    assert session.last_metrics.stage_boundary_batches == 1
    assert session.last_metrics.copied_batches == 1


def test_cpp_python_island_exception_does_not_poison_next_plan_run() -> None:
    """例外後の協調sessionを再利用せず、同じPlanを再実行できる。"""

    should_fail = True

    def fail_once(value: int) -> int:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("intentional Python Stage failure")
        return value * 2

    plan = cw.compile([cw.output(cw.Flow([2]).map(fail_once), collector=cw.Latest())])

    with pytest.raises(cw.KernelExecutionError, match="intentional Python Stage failure"):
        plan.run(executor="cpp")

    result = plan.run(executor="cpp")
    assert result.outputs[0].emissions[0].value == 4


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
    assert isinstance(session, CppSession)

    result = session.run()

    assert result == plan.run(executor=cw.PythonExecutor())
    assert result.outputs[0].emissions == ()
    assert result.outputs[0].received_count == 4
    assert session.last_metrics is not None
    assert session.last_metrics.output_boundary_bytes == 0
    assert session.last_metrics.owned_input_bytes > 0
    assert session.last_metrics.python_native_transitions == 2
    assert session.last_metrics.stage_python_dispatches == 0
    assert session.last_metrics.native_run_releases_gil
    assert session.last_metrics.python_free_hot_path
    assert session.last_metrics.execution_classification == "all_native"
    assert session.last_metrics.public_emission_reconstructions == 0
    assert session.last_metrics.python_boundary_dispatches == 0
    assert session.last_metrics.boundary_batch_conversions == 0


def test_cpp_cooperative_stage_session_requires_matching_resume() -> None:
    """C++状態機械がStage ID不一致とresume前のre-advanceを拒否する。"""

    session = CppCooperativeStageSession((3,))

    assert session.advance() == (0, 3)
    with pytest.raises(RuntimeError, match="contract=python_stage_resume_required"):
        session.advance()
    with pytest.raises(ValueError, match="contract=python_stage_resume_id"):
        session.resume(4)
    session.resume(3)
    assert session.advance() == (1, -1)


@pytest.mark.parametrize("sample_count", [8, 128])
def test_cpp_executor_python_boundary_work_is_bounded_by_collector(
    sample_count: int,
) -> None:
    """Emission数を増やしてもPython遷移と公開値復元をcollector上限でboundする。"""

    values = [(float(index), float(index)) for index in range(sample_count)]
    plan = _plan(
        collector=cw.Bounded(2, cw.OverflowPolicy.DROP_OLDEST),
        source_values=values,
    )
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert len(result.outputs[0].emissions) == 2
    assert isinstance(session, CppSession)
    assert session.last_metrics is not None
    assert session.last_metrics.python_native_transitions == 2
    assert session.last_metrics.public_emission_reconstructions == 2
    assert session.last_metrics.boundary_batch_conversions == 1
    assert session.last_metrics.python_boundary_dispatches == 0
    assert session.last_metrics.python_free_hot_path


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


def test_cpp_python_stage_continuous_session_is_explicitly_pending() -> None:
    """未実装の継続実行はPythonへ暗黙fallbackせず契約位置を報告する。"""

    mapped = cw.Flow([1, 2]).map(lambda value: value + 1)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    with pytest.raises(
        cw.SessionError,
        match="contract=python_stage_continuous_session_pending",
    ) as captured:
        plan.create_continuous_session(executor="cpp")

    message = str(captured.value)
    assert "stage=" in message
    assert "node=" in message
    assert "port=" in message
    assert "binding=" in message


def test_cpp_executor_continuous_session_advances_monotonically_and_drains() -> None:
    """有限native Planを論理時間境界ごとにC++で累積実行する。"""

    plan = _plan()
    session = plan.create_continuous_session(executor=cw.CppExecutor())

    assert session.state is cw.SessionState.CREATED
    session.start()
    assert session.run_until(Fraction(1, 1)).outputs[0].emissions == ()
    first_frame = session.run_until(2)
    assert first_frame.outputs[0].emissions == plan.run(duration=2).outputs[0].emissions
    with pytest.raises(cw.SessionError, match="strictly increasing"):
        session.run_until(2)
    closed = session.close()

    assert closed.outputs == plan.run().outputs
    assert closed.completed
    assert session.state is cw.SessionState.CLOSED


def test_cpp_executor_continuous_session_cancel_preserves_last_snapshot() -> None:
    """cancelはdrainせず直前snapshotと明示Diagnosticを保持する。"""

    session = _plan().create_continuous_session(executor="cpp")
    session.start()
    observed = session.run_until(Fraction(2, 1))

    cancelled = session.cancel()

    assert cancelled.outputs == observed.outputs
    assert not cancelled.completed
    assert cancelled.diagnostics[-1].code == "SESSION_CANCELLED"
    assert session.state is cw.SessionState.CANCELLED
    with pytest.raises(cw.SessionError, match="state=running"):
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
    assert isinstance(session, CppSession)
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

    session = plan.create_session(executor="cpp", extension_bindings={"record": extension})
    result = session.run()

    assert extension.sessions[0].events == [result.outputs[0].emissions[0]]
    assert extension.sessions[0].finalized
    assert isinstance(session, CppSession)
    assert session.last_metrics is not None
    assert session.last_metrics.python_free_hot_path
    assert session.last_metrics.public_emission_reconstructions == 4
    assert session.last_metrics.boundary_batch_conversions == 2
    assert session.last_metrics.python_boundary_dispatches == 3
