"""Kernel compile/run境界とPython Backendを定義する。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from .config import Config
from .model import LogicalInterval

T_co = TypeVar("T_co", covariant=True)


class GapPolicy(StrEnum):
    """入力gap境界後のstateful KernelState処理を表す。"""

    RESET = "reset"
    CONTINUE = "continue"


@dataclass(frozen=True)
class CallableAdapter:
    """plain callableへGraph上の実行契約を付与する。"""

    operation: Callable[..., object]
    max_items: int = 1
    accepts_invalid: bool = False
    time_transform: str = "preserve"
    gap_policy: GapPolicy = GapPolicy.RESET

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("callable adapter max_items must be positive")
        if self.time_transform not in {"preserve", "explicit"}:
            raise ValueError("callable adapter time_transform must be preserve or explicit")
        if not isinstance(self.gap_policy, GapPolicy):
            raise ValueError("gap_policy must be a GapPolicy")

    def __call__(self, value: object, **arguments: object) -> object:
        """元のcallableへ主入力と解決済みkeyword引数を渡す。"""

        return self.operation(value, **arguments)


def callable_kernel(
    operation: Callable[..., object],
    *,
    max_items: int = 1,
    accepts_invalid: bool = False,
    time_transform: str = "preserve",
    gap_policy: GapPolicy = GapPolicy.RESET,
) -> CallableAdapter:
    """plain callableを明示実行契約付きadapterへ変換する。

    Args:
        operation: 主入力とkeyword引数を受け取るcallable。
        max_items: 一回の呼出しで生成できるEmission上限。
        accepts_invalid: INVALID入力でも呼び出す場合にTrue。
        time_transform: 入力interval維持は`preserve`、Kernel明示変更は`explicit`。
        gap_policy: gap後にKernelStateをresetまたは継続する規則。

    Returns:
        Flow.mapへ渡せる不変CallableAdapter。

    Raises:
        ValueError: 件数、time transform、gap policyが不正な場合。
    """

    return CallableAdapter(
        operation,
        max_items,
        accepts_invalid,
        time_transform,
        gap_policy,
    )


@dataclass(frozen=True)
class CompileContext:
    """Kernel.compileへ渡す不変なNode設定を表す。

    Args:
        config: Nodeが参照する不変Config。
        constants: Flow.mapで固定したprocess-local定数。
        node_id: compile対象Node ID。legacy BackendではNoneを許可する。
        input_shapes: 各input Portの解決済みitem shape。Noneは静的証明不能。
        output_shapes: 各output Portの解決済みitem shape。Noneは静的証明不能。
        input_dtypes: 各input Portの解決済みdtype。
        output_dtypes: 各output Portの解決済みdtype。
        output_port_ids: output_shapesに対応するPort ID。

    境界条件:
        shapeはOperationSpecからCompilerが解決した値だけを渡し、Backendが
        runtime値から別のshape意味論を推測してはならない。
    """

    config: Config
    constants: Mapping[str, object]
    node_id: int | None = None
    input_shapes: tuple[tuple[int, ...] | None, ...] = ()
    output_shapes: tuple[tuple[int, ...] | None, ...] = ()
    input_dtypes: tuple[str | None, ...] = ()
    output_dtypes: tuple[str | None, ...] = ()
    output_port_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunContext:
    """KernelState.processへ渡す一回のEmission区間を表す。"""

    config: Config
    interval: LogicalInterval


@runtime_checkable
class KernelState(Protocol[T_co]):
    """一つのSessionに閉じた能動的なKernel実行状態のprotocol。

    実装は必要に応じて`flush()`と`close()`も持てる。v0.4 runtimeは`close()`を
    capability検出し、output-producing `flush()`はportable Emission ABI確定後に扱う。
    状態本体、workspace、native handleを所有し、別Sessionや例外後の再実行へ共有してはならない。
    """

    def process(self, inputs: tuple[object, ...], context: RunContext) -> T_co:
        """入力値列から一つのKernel戻り値を生成する。"""

        ...


@runtime_checkable
class Kernel(Protocol[T_co]):
    """Backendが生成する、複数Session間で共有可能な不変実装factory。"""

    def create_state(self) -> KernelState[T_co]:
        """一つのSessionだけが所有する空のKernelStateを生成する。"""

        ...


@runtime_checkable
class NativeKernel(Kernel[T_co], Protocol[T_co]):
    """PortablePlanIRへexport可能なnative Kernel ABI付きfactory。"""

    abi_version: str
    process_model: str
    workspace_size_bytes: int
    workspace_alignment_bytes: int
    supports_flush: bool
    session_local: bool
    native_compatible: bool


@runtime_checkable
class NativeBatchKernelState(KernelState[T_co], Protocol[T_co]):
    """固定shape item batchを一回のnative呼出しで処理するKernelState。"""

    def process_batch(
        self,
        values: memoryview[float],
        *,
        item_count: int,
        item_shape: tuple[int, ...],
    ) -> object:
        """read-only contiguous f64 batchを処理してnative batch結果を返す。"""

        ...


@runtime_checkable
class NativeBatchKernel(NativeKernel[T_co], Protocol[T_co]):
    """run-local batch KernelStateを生成できるnative Kernel factory。"""

    def create_state(self) -> NativeBatchKernelState[T_co]:
        """一つのSessionだけが所有するbatch対応KernelStateを生成する。"""

        ...


@runtime_checkable
class NativeValueSchemaProvider(Protocol):
    """入力固定shapeからnative出力shapeをcompile時に解決するprotocol。"""

    output_dtype: str

    def resolve_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """一つの入力item shapeから一つの出力item shapeを返す。"""

        ...


@dataclass(frozen=True)
class NativeKernelRuntimeBinding:
    """PortablePlanIRのKernel slotへ注入するprocess-local native定数。

    Args:
        abi_version: PortablePlanIRのKernel ABIと一致すべきversion付きID。
        process_model: native runtimeが選択する処理モデル。
        parameter_dtype: parameter bufferの要素型。
        parameter_shape: parameter bufferの固定shape。
        parameter_bytes: native endianで連続したimmutable parameter値。

    Raises:
        ValueError: ABI、dtype、shapeまたはbyte長が自己矛盾する場合。

    境界条件:
        pointerやallocatorは保持せず、binding自身がimmutable bytesを所有する。
    """

    abi_version: str
    process_model: str
    parameter_dtype: str
    parameter_shape: tuple[int, ...]
    parameter_bytes: bytes

    def __post_init__(self) -> None:
        if not self.abi_version or not self.process_model:
            raise ValueError("native Kernel runtime binding requires ABI and process model")
        if self.parameter_dtype != "float64":
            raise ValueError("native Kernel runtime binding currently requires float64 parameters")
        if any(item <= 0 for item in self.parameter_shape):
            raise ValueError("native Kernel runtime binding requires a non-negative fixed shape")
        element_count = 1 if self.parameter_shape else 0
        for item in self.parameter_shape:
            element_count *= item
        if len(self.parameter_bytes) != element_count * 8:
            raise ValueError("native Kernel parameter byte length does not match its shape")


@runtime_checkable
class NativeRuntimeBindingProvider(Protocol):
    """CppExecutor用のprocess-local Kernel bindingを生成するprotocol。"""

    def create_native_runtime_binding(self) -> NativeKernelRuntimeBinding:
        """pointerを含まないimmutable native定数bindingを返す。"""

        ...


@runtime_checkable
class KernelProvider(Protocol[T_co]):
    """legacy直接Kernel記述をcompile済みKernelへ変換する内部protocol。"""

    def compile(self, context: CompileContext) -> Kernel[T_co]:
        """Node固有の不変Kernelを一回生成する。"""

        ...


class Backend(Protocol):
    """KernelProviderを不変Kernelへ変換するBackend protocol。"""

    @property
    def name(self) -> str:
        """exportとDiagnosticに使用するBackend名を返す。"""

        ...

    def compile_kernel(
        self,
        kernel: KernelProvider[object],
        context: CompileContext,
    ) -> Kernel[object]:
        """指定KernelProviderをこのBackendでcompileする。"""

        ...


@dataclass(frozen=True)
class PythonBackend:
    """Kernel自身のPython compile実装を呼ぶv0.1 Backend。"""

    name: str = "python"

    def compile_kernel(
        self,
        kernel: KernelProvider[object],
        context: CompileContext,
    ) -> Kernel[object]:
        """Kernel.compileの結果をそのまま返す。"""

        return kernel.compile(context)


@dataclass(frozen=True)
class _PythonCallableState:
    """一つのSessionに閉じてPython callableを呼び出すKernelState。"""

    operation: Callable[..., object]
    constants: tuple[tuple[str, object], ...]
    input_keywords: tuple[str, ...]
    inject_config: bool

    def process(self, inputs: tuple[object, ...], context: RunContext) -> object:
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
class _PythonCallableKernel:
    """Python callableと解決済み引数を保持する不変Kernel。"""

    operation: Callable[..., object]
    constants: tuple[tuple[str, object], ...]
    input_keywords: tuple[str, ...]
    inject_config: bool

    def create_state(self) -> _PythonCallableState:
        """可変状態を共有しないPython callable KernelStateを生成する。"""

        return _PythonCallableState(
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

    def compile(self, context: CompileContext) -> Kernel[object]:
        """定数引数を固定し、run-local KernelState factoryを返す。"""

        return _PythonCallableKernel(
            self.operation,
            tuple(context.constants.items()),
            self.input_keywords,
            self.inject_config,
        )


# v0.4公開名からの一時的な型alias。内部実装は正式名称だけを使用する。
CompiledKernel = Kernel
CompiledKernelSession = KernelState
NativeCompiledKernel = NativeKernel
NativeBatchCompiledKernel = NativeBatchKernel
NativeBatchKernelSession = NativeBatchKernelState
