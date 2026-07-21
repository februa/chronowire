"""ExecutionPlanとExecutorの責務境界を検証する。"""

from collections.abc import Mapping

import pytest

import chronowire as cw


class _RecordingExecutor:
    """PythonExecutorへ委譲し、選択境界の通過回数だけを記録する。"""

    name = "recording-python"

    def __init__(self) -> None:
        self.one_shot_count = 0
        self.plan_session_count = 0
        self._delegate = cw.PythonExecutor()

    def create_session(
        self,
        plan: cw.ExecutionPlan,
        extension_bindings: Mapping[str, cw.Extension] | None,
    ) -> cw.ExecutionSession:
        """一回実行境界を記録してPythonExecutorへ委譲する。"""

        self.one_shot_count += 1
        return self._delegate.create_session(plan, extension_bindings)

    def create_plan_session(
        self,
        plan: cw.ExecutionPlan,
        extension_bindings: Mapping[str, cw.Extension] | None,
        options: cw.RuntimeOptions | None,
    ) -> cw.PlanSession:
        """継続実行境界を記録してPythonExecutorへ委譲する。"""

        self.plan_session_count += 1
        return self._delegate.create_plan_session(plan, extension_bindings, options)


def test_execution_plan_selects_executor_without_changing_trace() -> None:
    """Executor実体を差し込んでもPython基準意味論を維持する。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2]), collector=cw.Bounded(2))])
    executor = _RecordingExecutor()

    expected = plan.run()
    actual = plan.run(executor=executor)

    assert actual == expected
    assert executor.one_shot_count == 1


def test_plan_session_selects_executor_at_session_creation() -> None:
    """継続sessionも同じExecutor選択境界を通る。"""

    plan = cw.compile([cw.output(cw.Flow([1]), collector=cw.Latest())])
    executor = _RecordingExecutor()

    session = plan.create_plan_session(executor=executor)
    session.start()
    result = session.close()

    assert result.completed
    assert executor.plan_session_count == 1


def test_unknown_executor_is_rejected_before_runtime_state_creation() -> None:
    """未対応Executor名をPythonへ暗黙fallbackしない。"""

    plan = cw.compile([cw.Flow([1])])

    with pytest.raises(ValueError, match="unsupported executor"):
        plan.run(executor="native")
