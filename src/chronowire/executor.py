"""Planからrun-local sessionを生成するExecutor境界を定義する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .extension import Extension
    from .model import LogicalTime
    from .runtime import (
        ContinuousSession,
        Plan,
        RunResult,
        RuntimeOptions,
        Session,
        SessionState,
    )


@runtime_checkable
class SessionRunner(Protocol):
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
class ContinuousSessionRunner(Protocol):
    """継続実行Executor sessionの公開lifecycle操作。"""

    @property
    def state(self) -> SessionState:
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
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> SessionRunner:
        """一回実行用のrun-local sessionを生成する。"""

        ...

    def create_continuous_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ContinuousSessionRunner:
        """継続実行用のrun-local sessionを生成する。"""

        ...


@dataclass(frozen=True)
class PythonExecutor:
    """既存の単一thread決定的Schedulerを選択するExecutor。"""

    name: str = "python"

    def create_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> Session:
        """既存Python runtimeを所有する一回実行sessionを生成する。"""

        return plan._create_python_session(extension_bindings)

    def create_continuous_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ContinuousSession:
        """既存Python runtimeを所有する継続sessionを生成する。"""

        return plan._create_python_continuous_session(extension_bindings, options)


@dataclass(frozen=True)
class CythonExecutor:
    """schema 0.3の限定f64 Stageを選択するsemantic prototype。"""

    name: str = "cython"

    def create_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> SessionRunner:
        """STRICT検証済みのCython一回実行sessionを生成する。"""

        if extension_bindings:
            raise ValueError("CythonExecutor prototype does not support Extension bindings")
        from .cython_executor import CythonSession

        return CythonSession(plan)

    def create_continuous_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ContinuousSessionRunner:
        """未実装の継続Cython sessionを暗黙fallbackせず拒否する。"""

        del plan, extension_bindings, options
        from .errors import SessionError

        raise SessionError(
            "CythonExecutor prototype does not support ContinuousSession; "
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
        native_run_releases_gil: C++ data plane実行中にGILを解放する契約ならTrue。
        public_emission_reconstructions: native outputから公開Emissionを復元した件数。
        python_boundary_dispatches: Extension等のPython境界callbackを呼び出した回数。
        boundary_batch_conversions: native item batchをPython公開値へ変換したbatch数。
        gil_acquisitions: Python Stage実行のためadapterがGILを取得した回数。
        stage_boundary_batches: Python/native Stage境界を通過したbatch数。
        stage_boundary_bytes: Stage境界でborrowまたはcopyした値byte数。
        zero_copy_batches: read-only borrowまたはbuffer protocolでcopyしなかったbatch数。
        copied_batches: 契約不適合のため一回copyしたbatch数。
        python_stage_ns: Python island内で費やした時間。
        native_stage_ns: native Stage内で費やした時間。
        execution_classification: `all_native`、`hybrid`、`python_stage_dominated`の分類。

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
    native_run_releases_gil: bool = True
    public_emission_reconstructions: int = 0
    python_boundary_dispatches: int = 0
    boundary_batch_conversions: int = 0
    gil_acquisitions: int = 0
    stage_boundary_batches: int = 0
    stage_boundary_bytes: int = 0
    zero_copy_batches: int = 0
    copied_batches: int = 0
    python_stage_ns: int = 0
    native_stage_ns: int = 0
    execution_classification: str = "all_native"

    @property
    def python_free_hot_path(self) -> bool:
        """C++ data plane内にEmission単位のPython dispatchがないかを返す。

        Returns:
            native runがGILを解放し、Stage内Python dispatchが0ならTrue。

        境界条件:
            RunResult、collector、Extension境界での復元やcallbackはhot pathに
            含めず、別fieldで計数する。
        """

        return (
            self.execution_classification == "all_native"
            and self.native_run_releases_gil
            and self.stage_python_dispatches == 0
        )


@dataclass(frozen=True)
class CppExecutor:
    """compile済みPortablePlanIRをrun-local C++ runtimeで運用するExecutor。"""

    name: str = "cpp"

    def create_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> SessionRunner:
        """検証済みPlanから自立したC++一回実行sessionを生成する。

        Raises:
            ValueError: Extension bindingまたは最小C++契約外Planの場合。
        """

        from .cpp_executor import CppSession

        validated = plan._create_python_session(extension_bindings)
        if plan.portable_ir.stages and all(
            stage.execution_domain == "python" for stage in plan.portable_ir.stages
        ):
            from .cpp_executor import CppPythonStageSession

            return CppPythonStageSession(plan, validated)
        if any(stage.execution_domain == "python" for stage in plan.portable_ir.stages):
            from .cpp_executor import (
                CppMixedSession,
                CppMultiIslandSession,
                CppPythonPrefixSession,
            )

            python_stage_count = sum(
                stage.execution_domain == "python" for stage in plan.portable_ir.stages
            )
            if python_stage_count > 1:
                return CppMultiIslandSession(plan)
            if plan.portable_ir.stages[0].execution_domain == "python":
                return CppPythonPrefixSession(plan)
            return CppMixedSession(plan)
        return CppSession(plan, validated._extensions)

    def create_continuous_session(
        self,
        plan: Plan,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> ContinuousSessionRunner:
        """有限native Planから継続論理時間C++ sessionを生成する。"""

        from .cpp_executor import CppContinuousSession
        from .errors import SessionError

        python_stages = tuple(
            stage for stage in plan.portable_ir.stages if stage.execution_domain == "python"
        )
        if python_stages:
            stage = python_stages[0]
            node_id = stage.node_ids[-1]
            node = next(item for item in plan.portable_ir.nodes if item.node_id == node_id)
            binding = next(
                (item.slot_id for item in plan.portable_ir.bindings if item.node_id == node_id),
                None,
            )
            raise SessionError(
                "CppExecutor Python Stage ContinuousSession is not implemented; "
                f"stage={stage.stage_id} node={node_id} port={node.output_port_ids[-1]} "
                f"binding={binding}; contract=python_stage_continuous_session_pending"
            )

        validated = plan._create_python_session(extension_bindings)
        return CppContinuousSession(plan, options, validated._extensions)
