"""Kernel compile/run境界とPython Backendを定義する。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from .config import Config
from .model import LogicalInterval

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class CompileContext:
    """Kernel.compileへ渡す不変なNode設定を表す。"""

    config: Config
    constants: Mapping[str, object]


@dataclass(frozen=True)
class RunContext:
    """CompiledKernel.runへ渡す一回のEmission区間を表す。"""

    config: Config
    interval: LogicalInterval


@runtime_checkable
class CompiledKernelSession(Protocol[T_co]):
    """一回のExecutionPlan.runに閉じたKernel実行状態のprotocol。"""

    def run(self, inputs: tuple[object, ...], context: RunContext) -> T_co:
        """入力値列から一つのKernel戻り値を生成する。"""

        ...


@runtime_checkable
class CompiledKernel(Protocol[T_co]):
    """Backendが生成する、複数run間で共有可能なKernel factoryのprotocol。"""

    def create_session(self) -> CompiledKernelSession[T_co]:
        """一回のrunだけが所有する空の実行sessionを生成する。"""

        ...


@runtime_checkable
class Kernel(Protocol[T_co]):
    """Config解決と作業領域準備をrunから分離するprotocol。"""

    def compile(self, context: CompileContext) -> CompiledKernel[T_co]:
        """Node固有のCompiledKernelを一回生成する。"""

        ...


class Backend(Protocol):
    """KernelをCompiledKernelへ変換するBackend protocol。"""

    @property
    def name(self) -> str:
        """exportとDiagnosticに使用するBackend名を返す。"""

        ...

    def compile_kernel(
        self,
        kernel: Kernel[object],
        context: CompileContext,
    ) -> CompiledKernel[object]:
        """指定KernelをこのBackendでcompileする。"""

        ...


@dataclass(frozen=True)
class PythonBackend:
    """Kernel自身のPython compile実装を呼ぶv0.1 Backend。"""

    name: str = "python"

    def compile_kernel(
        self,
        kernel: Kernel[object],
        context: CompileContext,
    ) -> CompiledKernel[object]:
        """Kernel.compileの結果をそのまま返す。"""

        return kernel.compile(context)


@dataclass(frozen=True)
class _PythonCallableSession:
    """一回のrunに閉じてPython callableを呼び出す内部session。"""

    operation: Callable[..., object]
    constants: tuple[tuple[str, object], ...]
    input_keywords: tuple[str, ...]
    inject_config: bool

    def run(self, inputs: tuple[object, ...], context: RunContext) -> object:
        """主入力と追加Flow入力を元のPython callableへ渡す。"""

        if not inputs:
            raise ValueError("Python callable Kernel requires a primary input")
        if len(inputs) - 1 != len(self.input_keywords):
            raise ValueError("Python callable Kernel input count does not match its Graph edges")
        arguments = dict(self.constants)
        arguments.update(zip(self.input_keywords, inputs[1:], strict=True))
        if self.inject_config:
            arguments["config"] = context.config
        return self.operation(inputs[0], **arguments)


@dataclass(frozen=True)
class _CompiledPythonCallable:
    """Python callableと解決済み引数を保持する内部session factory。"""

    operation: Callable[..., object]
    constants: tuple[tuple[str, object], ...]
    input_keywords: tuple[str, ...]
    inject_config: bool

    def create_session(self) -> _PythonCallableSession:
        """可変状態を共有しないPython callable sessionを生成する。"""

        return _PythonCallableSession(
            self.operation,
            self.constants,
            self.input_keywords,
            self.inject_config,
        )


@dataclass(frozen=True)
class PythonCallableKernel:
    """Python callableを通常のKernel lifecycleへ正規化する内部Kernel。"""

    operation: Callable[..., object]
    input_keywords: tuple[str, ...]
    inject_config: bool = False

    def compile(self, context: CompileContext) -> CompiledKernel[object]:
        """定数引数を固定し、run-local session factoryを返す。"""

        return _CompiledPythonCallable(
            self.operation,
            tuple(context.constants.items()),
            self.input_keywords,
            self.inject_config,
        )
