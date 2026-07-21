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
