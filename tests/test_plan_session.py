"""v0.2 PlanSessionの継続状態とlifecycle契約を検証する。"""

from __future__ import annotations

from fractions import Fraction

import pytest

import chronowire as cw


class _CounterSession:
    """PlanSession内でだけ状態を保持するtest session。"""

    def __init__(self) -> None:
        self._count = 0

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        """入力ごとにsession-local counterを進める。"""

        self._count += 1
        return self._count


class _CompiledCounter:
    """独立したcounter sessionを生成するcompile済みKernel。"""

    def create_session(self) -> _CounterSession:
        """counter=0の新しいsessionを返す。"""

        return _CounterSession()


class _CounterKernel:
    """継続状態の境界試験に使うtest Kernel。"""

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        """共有可能なcompile済みcounter factoryを返す。"""

        return _CompiledCounter()


class _MaybeFailSession:
    """指定されたsessionだけKernel実行を失敗させるtest session。"""

    def __init__(self, should_fail: bool) -> None:
        self._should_fail = should_fail

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        """失敗対象でなければ入力をそのまま返す。"""

        if self._should_fail:
            raise RuntimeError("planned failure")
        return inputs[0]


class _FailFirstCompiled:
    """最初のrun-local sessionだけ失敗させるcompile済みKernel。"""

    def __init__(self) -> None:
        self._created = 0

    def create_session(self) -> _MaybeFailSession:
        """一回目だけ失敗する独立sessionを生成する。"""

        self._created += 1
        return _MaybeFailSession(self._created == 1)


class _FailFirstKernel:
    """失敗後のExecutionPlan再利用を検証するtest Kernel。"""

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        """失敗回数をcompile済みfactoryだけで管理する。"""

        return _FailFirstCompiled()


def _values(result: cw.RunResult) -> list[object]:
    return [item.value for item in result.outputs[0].emissions]


def test_plan_session_preserves_frame_state_across_run_until_boundaries() -> None:
    """境界外Source値を失わず、FRAME historyを次の呼出しへ保持する。"""

    frames = cw.Flow([1, 2, 3, 4]).frame(2)
    plan = cw.compile([cw.output(frames, collector=cw.Bounded(2))])
    session = plan.create_plan_session()

    assert session.state is cw.PlanSessionState.CREATED
    session.start()
    assert session.state is cw.PlanSessionState.RUNNING

    first = session.run_until(1)
    second = session.run_until(2)
    final = session.run_until(cw.LogicalTime(4))

    assert _values(first) == []
    assert not first.completed
    assert _values(second) == [(1, 2)]
    assert not second.completed
    assert _values(final) == [(1, 2), (3, 4)]
    assert final.completed

    closed = session.close()
    assert _values(closed) == [(1, 2), (3, 4)]
    assert closed.completed
    assert session.state is cw.PlanSessionState.CLOSED


def test_plan_session_preserves_kernel_state_but_new_session_resets_it() -> None:
    """同一sessionではKernel状態を継続し、別sessionでは初期化する。"""

    counted = cw.Flow([10, 20]).map(_CounterKernel())
    plan = cw.compile([cw.output(counted, collector=cw.Bounded(2))])

    first_session = plan.create_plan_session()
    first_session.start()
    assert _values(first_session.run_until(Fraction(1))) == [1]
    assert _values(first_session.run_until(Fraction(2))) == [1, 2]
    first_session.close()

    second_session = plan.create_plan_session()
    second_session.start()
    assert _values(second_session.run_until(1)) == [1]
    second_session.cancel()


def test_plan_session_requires_valid_lifecycle_and_monotonic_boundary() -> None:
    """未開始、二重開始、非単調境界、終了後操作を明示例外にする。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2]), collector=cw.Bounded(2))])
    session = plan.create_plan_session()

    with pytest.raises(cw.PlanSessionError, match="requires state=running"):
        session.run_until(1)
    session.start()
    with pytest.raises(cw.PlanSessionError, match="requires state=created"):
        session.start()
    session.run_until(1)
    with pytest.raises(cw.PlanSessionError, match="strictly increasing"):
        session.run_until(1)

    cancelled = session.cancel()
    assert session.state is cw.PlanSessionState.CANCELLED
    assert not cancelled.completed
    assert any(item.code == "SESSION_CANCELLED" for item in cancelled.diagnostics)
    with pytest.raises(cw.PlanSessionError, match="actual=cancelled"):
        session.close()


def test_plan_session_flushes_finite_source_after_partial_run() -> None:
    """finite Sourceの残りとpad_end FRAMEをflushでdrainする。"""

    frames = cw.Flow([1, 2, 3]).frame(2, pad_end=True)
    plan = cw.compile([cw.output(frames, collector=cw.Bounded(2))])
    session = plan.create_plan_session()
    session.start()

    assert _values(session.run_until(1)) == []
    flushed = session.flush()

    assert _values(flushed) == [(1, 2), (3, None)]
    assert flushed.completed
    closed = session.close()
    assert _values(closed) == [(1, 2), (3, None)]


def test_failed_plan_session_does_not_poison_execution_plan() -> None:
    """Kernel失敗sessionを閉じ、同じPlanから新しい状態で再実行できる。"""

    mapped = cw.Flow([7]).map(_FailFirstKernel())
    plan = cw.compile([cw.output(mapped, collector=cw.Latest())])
    failed = plan.create_plan_session()
    failed.start()

    with pytest.raises(cw.KernelExecutionError, match="planned failure"):
        failed.run_until(1)
    assert failed.state is cw.PlanSessionState.FAILED

    recovered = plan.create_plan_session()
    recovered.start()
    assert _values(recovered.run_until(1)) == [7]
    assert recovered.close().completed


def test_one_shot_duration_keeps_v01_completion_semantics() -> None:
    """既存run(duration)は境界到達を正常完了として扱う。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2]), collector=cw.Bounded(2))])

    result = plan.run(duration=1)

    assert _values(result) == [1]
    assert result.completed


def test_runtime_options_budget_allows_retrying_same_logical_boundary() -> None:
    """budget終了時は同じrun_until境界を再指定して継続できる。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2, 3]), collector=cw.Bounded(3))])
    session = plan.create_plan_session(options=cw.RuntimeOptions(max_scheduler_steps=1))
    session.start()

    first = session.run_until(3)
    second = session.run_until(3)

    assert _values(first) == [1]
    assert _values(second) == [1, 2]
    assert [item.code for item in second.diagnostics].count("EXECUTION_BUDGET_EXHAUSTED") == 2
    session.cancel()


def test_runtime_options_validates_watermarks_and_budget() -> None:
    """無効なchunk、watermark、budgetをGraph実行前に拒否する。"""

    with pytest.raises(ValueError, match="source_chunk_duration"):
        cw.RuntimeOptions(source_chunk_duration=Fraction(0))
    with pytest.raises(ValueError, match="requires port_high_watermark"):
        cw.RuntimeOptions(port_low_watermark=0)
    with pytest.raises(ValueError, match="below port_high_watermark"):
        cw.RuntimeOptions(port_high_watermark=2, port_low_watermark=2)
    with pytest.raises(ValueError, match="max_scheduler_steps"):
        cw.RuntimeOptions(max_scheduler_steps=0)

    source = cw.Flow([1, 2])
    framed = source.frame(2)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)
    session = cw.compile([merged]).create_plan_session(
        options=cw.RuntimeOptions(port_high_watermark=1)
    )
    with pytest.raises(cw.PlanSessionError, match="below compiled capacity"):
        session.start()
