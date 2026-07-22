"""ExecutionPlanからrun-local sessionを生成するExecutor境界を定義する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .extension import Extension
    from .model import LogicalTime
    from .runtime import (
        ExecutionPlan,
        ExecutionSession,
        PlanSession,
        PlanSessionState,
        RunResult,
        RuntimeOptions,
    )


@runtime_checkable
class ExecutorSession(Protocol):
    """一回実行Executor sessionの公開操作。"""

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """run-local状態でPlanを一回実行する。"""

        ...


@runtime_checkable
class ExecutorPlanSession(Protocol):
    """継続実行Executor sessionの公開lifecycle操作。"""

    @property
    def state(self) -> PlanSessionState:
        """現在のsession状態を返す。"""

        ...

    def start(self) -> None:
        """run-local resourceを生成して開始する。"""

        ...

    def run_until(self, logical_end: LogicalTime | Fraction | int | float) -> RunResult:
        """状態を保持したまま指定論理時刻まで進める。"""

        ...

    def flush(self) -> RunResult:
        """有限入力とpending frameをdrainする。"""

        ...

    def close(self) -> RunResult:
        """入力を停止してdrain後にresourceを解放する。"""

        ...

    def cancel(self) -> RunResult:
        """pending値を破棄してresourceを解放する。"""

        ...


@runtime_checkable
class Executor(Protocol):
    """PortablePlanIRの意味論から実行sessionを生成するprotocol。"""

    @property
    def name(self) -> str:
        """Plan exportとDiagnosticに使用するExecutor名を返す。"""

        ...

    def create_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> ExecutorSession:
        """一回実行用のrun-local sessionを生成する。"""

        ...

    def create_plan_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ExecutorPlanSession:
        """継続実行用のrun-local sessionを生成する。"""

        ...


@dataclass(frozen=True)
class PythonExecutor:
    """既存の単一thread決定的Schedulerを選択するExecutor。"""

    name: str = "python"

    def create_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> ExecutionSession:
        """既存Python runtimeを所有する一回実行sessionを生成する。"""

        return plan._create_python_session(extension_bindings)

    def create_plan_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> PlanSession:
        """既存Python runtimeを所有する継続sessionを生成する。"""

        return plan._create_python_plan_session(extension_bindings, options)


@dataclass(frozen=True)
class CythonExecutor:
    """schema 0.3の限定f64 Stageを選択するsemantic prototype。"""

    name: str = "cython"

    def create_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> ExecutorSession:
        """STRICT検証済みのCython一回実行sessionを生成する。"""

        if extension_bindings:
            raise ValueError("CythonExecutor prototype does not support Extension bindings")
        from .cython_executor import CythonExecutionSession

        return CythonExecutionSession(plan)

    def create_plan_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ExecutorPlanSession:
        """未実装の継続Cython sessionを暗黙fallbackせず拒否する。"""

        del plan, extension_bindings, options
        from .errors import PlanSessionError

        raise PlanSessionError(
            "CythonExecutor prototype does not support PlanSession; "
            "contract=cython_continuous_session"
        )


@dataclass(frozen=True)
class CppRuntimeMetrics:
    """一回のC++ runtime実行で観測したnative境界指標。

    Args:
        scheduler_ns: RATEとFRAMEを含むC++ Scheduler時間。
        kernel_ns: C++固定CBF処理時間。
        output_select_ns: native collector policy適用時間。
        owned_input_bytes: sessionが所有するSource、時刻、status、Kernel定数byte。
        output_boundary_bytes: Python観測境界へcopyした値byte。
        python_native_transitions: 一回のrunでPython/native境界を跨ぐ回数。
        stage_python_dispatches: native Stage内のPython method dispatch数。
        executed_node_count: fan-outを含む一回のrunで評価したNode数。

    境界条件:
        Pythonでの公開Emission復元時間とobject memoryは含まない。
    """

    scheduler_ns: int
    kernel_ns: int
    output_select_ns: int
    owned_input_bytes: int
    output_boundary_bytes: int
    python_native_transitions: int = 2
    stage_python_dispatches: int = 0
    executed_node_count: int = 0


@dataclass(frozen=True)
class CppExecutor:
    """compile済みPortablePlanIRをrun-local C++ runtimeで運用するExecutor。"""

    name: str = "cpp"

    def create_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> ExecutorSession:
        """検証済みPlanから自立したC++一回実行sessionを生成する。

        Raises:
            ValueError: Extension bindingまたは最小C++契約外Planの場合。
        """

        from .cpp_executor import CppExecutionSession

        validated = plan._create_python_session(extension_bindings)
        return CppExecutionSession(plan, validated._extensions)

    def create_plan_session(
        self,
        plan: ExecutionPlan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ExecutorPlanSession:
        """有限native Planから継続論理時間C++ sessionを生成する。"""

        from .cpp_executor import CppPlanSession

        validated = plan._create_python_session(extension_bindings)
        return CppPlanSession(plan, options, validated._extensions)
