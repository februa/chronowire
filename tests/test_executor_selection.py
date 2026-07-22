"""PlanとExecutorの責務境界を検証する。"""

import pytest

import chronowire as cw
from chronowire.executor import IncrementalSessionRunner, RunSessionRunner
from chronowire.runtime import _BoundExtension


class _RecordingExecutor:
    """PythonExecutorへ委譲し、選択境界の通過回数だけを記録する。"""

    name = "recording-python"

    def __init__(self) -> None:
        self.one_shot_count = 0
        self.incremental_count = 0
        self._delegate = cw.PythonExecutor()

    def _create_run_session(
        self,
        plan: cw.Plan,
        extensions: tuple[_BoundExtension, ...],
    ) -> RunSessionRunner:
        """一回実行境界を記録してPythonExecutorへ委譲する。"""

        self.one_shot_count += 1
        return self._delegate._create_run_session(plan, extensions)

    def _create_incremental_session(
        self,
        plan: cw.Plan,
        extensions: tuple[_BoundExtension, ...],
        options: cw.RuntimeOptions | None,
    ) -> IncrementalSessionRunner:
        """段階実行境界を記録してPythonExecutorへ委譲する。"""

        self.incremental_count += 1
        return self._delegate._create_incremental_session(plan, extensions, options)


def test_plan_selects_executor_without_changing_trace() -> None:
    """Executor実体を差し込んでもPython基準意味論を維持する。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2]), collector=cw.Bounded(2))])
    executor = _RecordingExecutor()

    expected = plan.run()
    actual = plan.run(executor=executor)

    assert actual == expected
    assert executor.one_shot_count == 1


def test_incremental_session_selects_executor_when_started() -> None:
    """段階実行も同じSessionからExecutor境界を通る。"""

    plan = cw.compile([cw.output(cw.Flow([1]), collector=cw.Latest())])
    executor = _RecordingExecutor()

    session = plan.create_session(executor=executor)
    session.start()
    result = session.close()

    assert result.completed
    assert executor.incremental_count == 1


def test_unknown_executor_is_rejected_before_runtime_state_creation() -> None:
    """未対応Executor名をPythonへ暗黙fallbackしない。"""

    plan = cw.compile([cw.Flow([1])])

    with pytest.raises(ValueError, match="unsupported executor"):
        plan.run(executor="native")
