"""process-localなC ABI Operation moduleを読込みBackendへ接続する。"""

from __future__ import annotations

import ctypes
from array import array
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import MappingProxyType

from .config import ConfigView
from .errors import MissingImplementationError
from .kernel import (
    CompileContext,
    Kernel,
    KernelState,
    RunContext,
)
from .model import Diagnostic, Emission, EmissionStatus, Severity
from .operation import ImplementationBinding, ImplementationSpec, OperationSpec

MODULE_ABI_V1 = "chronowire.operation-module.v1"


def native_operation_include_dir() -> Path:
    """C/C++ Operation wrapper向け公開ABI headerのinclude directoryを返す。

    Returns:
        `native_operation_abi.h`を含むインストール済みpackage directory。

    Raises:
        RuntimeError: packageが壊れ、ABI headerが配布物に含まれない場合。

    境界条件:
        返すpathはbuild時のcontrol-plane情報であり、PortablePlanIRへ保存しない。
        DSP本体ではなくOperation wrapperだけがこのheaderに依存する。
    """

    include_dir = Path(__file__).resolve().parent
    header = include_dir / "native_operation_abi.h"
    if not header.is_file():
        raise RuntimeError(f"Chronowire native Operation ABI header is missing: {header}")
    return include_dir


class _COperationEntryV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("operation_id", ctypes.c_char_p),
        ("implementation_id", ctypes.c_char_p),
        ("abi_version", ctypes.c_char_p),
        ("process_model", ctypes.c_char_p),
        ("workspace_size_bytes", ctypes.c_size_t),
        ("workspace_alignment_bytes", ctypes.c_size_t),
        ("flags", ctypes.c_uint32),
        ("create", ctypes.c_void_p),
        ("process", ctypes.c_void_p),
        ("flush", ctypes.c_void_p),
        ("destroy", ctypes.c_void_p),
    ]


class _COperationModuleV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("module_abi_version", ctypes.c_char_p),
        ("operation_count", ctypes.c_size_t),
        ("operations", ctypes.POINTER(_COperationEntryV1)),
    ]


class _CBufferViewV1(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("value_count", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
    ]


class _CMutableBufferViewV1(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("value_capacity", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
    ]


class _CProcessResultV1(ctypes.Structure):
    _fields_ = [
        ("output_count", ctypes.c_size_t),
        ("status", ctypes.c_uint8),
        ("diagnostic_severity", ctypes.c_uint8),
        ("diagnostic_code", ctypes.c_char_p),
        ("diagnostic_message", ctypes.c_char_p),
    ]


_CreateFunctionV1 = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)
_ProcessFunctionV1 = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_CBufferViewV1),
    ctypes.c_size_t,
    ctypes.POINTER(_CMutableBufferViewV1),
    ctypes.POINTER(_CProcessResultV1),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)
_DestroyFunctionV1 = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _text(value: bytes | None, field: str) -> str:
    if value is None:
        raise ValueError(f"native module manifest field {field} must not be null")
    try:
        result = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"native module manifest field {field} must be UTF-8") from error
    if not result:
        raise ValueError(f"native module manifest field {field} must not be empty")
    return result


@dataclass(frozen=True)
class NativeOperationEntry:
    """共有library内の一Operation entryをprocess-localに保持する。

    Args:
        operation_id: OperationSpecと照合する安定ID。
        implementation_id: module内のversion付き実装ID。
        abi_version: process呼出しABI ID。
        process_model: 一回のprocessが扱うitem規則。
        workspace_size_bytes: 宣言workspace byte数。
        workspace_alignment_bytes: workspace alignment。
        supports_flush: flush entryを持つ場合にTrue。
        session_local: stateをrun-local sessionへ閉じる場合にTrue。
        create_address: create function address。
        process_address: process function address。
        flush_address: 任意flush function address。
        destroy_address: destroy function address。

    境界条件:
        pointerとmodule handleはPortablePlanIRへ保存されず、bindしたprocess内だけで有効である。
    """

    operation_id: str
    implementation_id: str
    abi_version: str
    process_model: str
    workspace_size_bytes: int
    workspace_alignment_bytes: int
    supports_flush: bool
    session_local: bool
    create_address: int
    process_address: int
    flush_address: int
    destroy_address: int
    _module: NativeOperationModule

    def implementation_spec(self, backend: str) -> ImplementationSpec:
        """portableな実装選択情報を返す。

        Args:
            backend: Planへ記録するBackend名。

        Returns:
            pointerとlibrary pathを含まないImplementationSpec。
        """

        return ImplementationSpec(
            self.operation_id,
            self.implementation_id,
            backend,
            self.abi_version,
            True,
            self.process_model,
            "external_module",
            (),
            self.workspace_size_bytes,
            self.workspace_alignment_bytes,
            self.supports_flush,
            self.session_local,
        )


class NativeOperationModule:
    """version付きC ABI module tableと共有library lifetimeを所有する。

    Args:
        path: `chronowire_operation_module_v1`をexportする共有library。

    Raises:
        OSError: libraryをloadできない場合。
        ValueError: symbol、module ABI、entry ID、function pointerが不正な場合。

    境界条件:
        path、CDLL handle、function pointerはprocess-localでありPortablePlanIRへ入れない。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._library = ctypes.CDLL(str(self._path))
        try:
            manifest = self._library.chronowire_operation_module_v1
        except AttributeError as error:
            raise ValueError(
                f"native module {self._path} lacks symbol chronowire_operation_module_v1"
            ) from error
        manifest.argtypes = []
        manifest.restype = ctypes.POINTER(_COperationModuleV1)
        pointer = manifest()
        if not pointer:
            raise ValueError(f"native module {self._path} returned a null manifest")
        table = pointer.contents
        if table.struct_size < ctypes.sizeof(_COperationModuleV1):
            raise ValueError("native module manifest struct_size is incompatible")
        if _text(table.module_abi_version, "module_abi_version") != MODULE_ABI_V1:
            raise ValueError("native module ABI does not match chronowire.operation-module.v1")
        if table.operation_count and not table.operations:
            raise ValueError("native module operation table is null")
        entries: dict[str, NativeOperationEntry] = {}
        for index in range(table.operation_count):
            item = table.operations[index]
            if item.struct_size < ctypes.sizeof(_COperationEntryV1):
                raise ValueError(f"native module entry index={index} has incompatible struct_size")
            operation_id = _text(item.operation_id, "operation_id")
            if operation_id in entries:
                raise ValueError(f"native module has duplicate operation_id={operation_id!r}")
            if not item.create or not item.process or not item.destroy:
                raise ValueError(
                    f"native module operation={operation_id} requires create/process/destroy"
                )
            supports_flush = bool(item.flags & 1)
            if item.flags & ~3:
                raise ValueError(
                    f"native module operation={operation_id} has unknown flags={item.flags}"
                )
            if supports_flush != bool(item.flush):
                raise ValueError(
                    f"native module operation={operation_id} flush flag/pointer mismatch"
                )
            if item.workspace_alignment_bytes == 0:
                raise ValueError(
                    f"native module operation={operation_id} requires positive alignment"
                )
            entries[operation_id] = NativeOperationEntry(
                operation_id,
                _text(item.implementation_id, "implementation_id"),
                _text(item.abi_version, "abi_version"),
                _text(item.process_model, "process_model"),
                item.workspace_size_bytes,
                item.workspace_alignment_bytes,
                supports_flush,
                bool(item.flags & 2),
                int(item.create),
                int(item.process),
                0 if item.flush is None else int(item.flush),
                int(item.destroy),
                self,
            )
        self._entries = MappingProxyType(entries)

    @property
    def path(self) -> Path:
        """load済み共有libraryの絶対pathを返す。

        Returns:
            process-localに解決した絶対path。IRへは保存しない。
        """

        return self._path

    @property
    def operations(self) -> Mapping[str, NativeOperationEntry]:
        """operation IDからmanifest entryへの読み取り専用mappingを返す。

        Returns:
            module lifetimeを保持するentry mapping。
        """

        return self._entries

    def binding(self, operation_id: str, *, backend: str = "cpp") -> ImplementationBinding:
        """別processでPortablePlanIRを復元するためのbindingを返す。

        Args:
            operation_id: moduleから選択するOperation ID。
            backend: ImplementationSpecへ記録するBackend名。

        Returns:
            module entryをprocess-local実体とするImplementationBinding。

        Raises:
            MissingImplementationError: moduleにoperation IDがない場合。
        """

        entry = self._entries.get(operation_id)
        if entry is None:
            raise MissingImplementationError(
                f"operation={operation_id} module={self._path} contract=missing_implementation"
            )
        return ImplementationBinding(entry.implementation_spec(backend), entry)


@dataclass(frozen=True)
class NativeOperationRuntimeBinding:
    """CppExecutorへmodule function tableを注入するprocess-local binding。

    Args:
        abi_version: OperationDescriptorと照合するABI ID。
        process_model: module entryのprocess model。
        parameter_dtype: v1では`float64`。
        parameter_shape: flatten済みConfig parameter shape。
        parameter_bytes: native endianのimmutable parameter bytes。
        create_address: create function address。
        process_address: process function address。
        flush_address: 任意flush function address。
        destroy_address: destroy function address。
        module: function addressのlifetimeを所有するmodule。

    Raises:
        ValueError: parameter、function table、module lifetimeが不整合な場合。

    境界条件:
        module handleとfunction addressはPortablePlanIRへ保存しない。
    """

    abi_version: str
    process_model: str
    parameter_dtype: str
    parameter_shape: tuple[int, ...]
    parameter_bytes: bytes
    create_address: int
    process_address: int
    flush_address: int
    destroy_address: int
    module: NativeOperationModule | None

    def __post_init__(self) -> None:
        if not self.abi_version or not self.process_model:
            raise ValueError("native Operation binding requires ABI and process model")
        if self.parameter_dtype != "float64":
            raise ValueError("native Operation binding currently requires float64 parameters")
        element_count = 1 if self.parameter_shape else 0
        for extent in self.parameter_shape:
            if extent <= 0:
                raise ValueError("native Operation parameter shape must be positive")
            element_count *= extent
        if len(self.parameter_bytes) != element_count * 8:
            raise ValueError("native Operation parameter bytes do not match their shape")
        if not self.create_address or not self.process_address or not self.destroy_address:
            raise ValueError("native Operation binding requires create/process/destroy pointers")
        if self.module is None:
            raise ValueError("native Operation binding must retain its module handle")


def _flatten_f64(value: object, shape: tuple[int, ...], *, input_index: int) -> tuple[float, ...]:
    """Python値を検証済みfixed shapeの連続float64列へ変換する。"""

    def visit(current: object, remaining: tuple[int, ...], path: tuple[int, ...]) -> list[float]:
        if not remaining:
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                raise TypeError(
                    f"input={input_index} index={path} expected=float64 actual={current!r}; "
                    "contract=native_f64_value"
                )
            return [float(current)]
        if not isinstance(current, (tuple, list)) or len(current) != remaining[0]:
            raise ValueError(
                f"input={input_index} index={path} expected_extent={remaining[0]} "
                f"actual={current!r}; contract=native_fixed_shape"
            )
        result: list[float] = []
        for index, item in enumerate(current):
            result.extend(visit(item, remaining[1:], (*path, index)))
        return result

    return tuple(visit(value, shape, ()))


def _reshape_f64(values: tuple[float, ...], shape: tuple[int, ...], offset: int = 0) -> object:
    """連続float64列をOperationSpecのitem shapeへ戻す。"""

    if not shape:
        return values[offset]
    child_width = prod(shape[1:])
    return tuple(
        _reshape_f64(values, shape[1:], offset + index * child_width) for index in range(shape[0])
    )


class _NativeModuleKernelState:
    """PythonExecutorでC ABI wrapperをEmission単位に照合するrun-local KernelState。"""

    def __init__(self, kernel: _NativeModuleKernel) -> None:
        self._compiled = kernel
        self._create = _CreateFunctionV1(kernel.entry.create_address)
        self._process = _ProcessFunctionV1(kernel.entry.process_address)
        self._destroy = _DestroyFunctionV1(kernel.entry.destroy_address)
        self._handle: int | None = None
        parameters = (ctypes.c_double * len(kernel.parameters))(*kernel.parameters)
        error = ctypes.create_string_buffer(512)
        handle = self._create(parameters, len(kernel.parameters), error, len(error))
        if not handle:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"node={kernel.node_id} port={kernel.output_port_id} "
                f"operation={kernel.entry.operation_id} error={message}; "
                "contract=native_module_create"
            )
        self._handle = int(handle)

    def process(self, inputs: tuple[object, ...], context: RunContext) -> object:
        """Operationの物理input列をC ABI wrapperで一回処理する。"""

        if self._handle is None:
            raise RuntimeError(
                f"operation={self._compiled.entry.operation_id} session is closed; "
                "contract=kernel_state_lifecycle"
            )
        if len(inputs) != len(self._compiled.input_shapes):
            raise ValueError(
                f"node={self._compiled.node_id} operation={self._compiled.entry.operation_id} "
                f"expected_inputs={len(self._compiled.input_shapes)} actual={len(inputs)}"
            )
        value_buffers: list[object] = []
        shape_buffers: list[object] = []
        views: list[_CBufferViewV1] = []
        for input_index, (value, shape) in enumerate(
            zip(inputs, self._compiled.input_shapes, strict=True)
        ):
            flat = _flatten_f64(value, shape, input_index=input_index)
            value_buffer = (ctypes.c_double * len(flat))(*flat)
            shape_buffer = (ctypes.c_size_t * len(shape))(*shape)
            value_buffers.append(value_buffer)
            shape_buffers.append(shape_buffer)
            views.append(_CBufferViewV1(value_buffer, len(flat), shape_buffer, len(shape)))
        input_views = (_CBufferViewV1 * len(views))(*views)
        output_count = prod(self._compiled.output_shape)
        output_values = (ctypes.c_double * output_count)()
        output_shape = (ctypes.c_size_t * len(self._compiled.output_shape))(
            *self._compiled.output_shape
        )
        output_view = _CMutableBufferViewV1(
            output_values,
            output_count,
            output_shape,
            len(self._compiled.output_shape),
        )
        process_result = _CProcessResultV1()
        error = ctypes.create_string_buffer(512)
        status = self._process(
            self._handle,
            input_views,
            len(views),
            ctypes.byref(output_view),
            ctypes.byref(process_result),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"node={self._compiled.node_id} port={self._compiled.output_port_id} "
                f"operation={self._compiled.entry.operation_id} error={message}; "
                "contract=native_module_process"
            )
        if (
            process_result.output_count != output_count
            or process_result.status > 2
            or process_result.diagnostic_severity > 2
            or output_view.value_capacity != output_count
            or output_view.rank != len(self._compiled.output_shape)
            or tuple(output_view.shape[index] for index in range(output_view.rank))
            != self._compiled.output_shape
        ):
            raise RuntimeError(
                f"node={self._compiled.node_id} port={self._compiled.output_port_id} "
                f"operation={self._compiled.entry.operation_id}; "
                "contract=native_module_result"
            )
        code = process_result.diagnostic_code
        message = process_result.diagnostic_message
        if (code is None) != (message is None) or bool(code) != bool(message):
            raise RuntimeError(
                f"node={self._compiled.node_id} port={self._compiled.output_port_id} "
                f"operation={self._compiled.entry.operation_id}; "
                "contract=native_module_diagnostic"
            )
        diagnostics = ()
        if code and message:
            diagnostics = (
                Diagnostic(
                    (Severity.INFO, Severity.WARNING, Severity.ERROR)[
                        process_result.diagnostic_severity
                    ],
                    code.decode("utf-8", errors="replace"),
                    message.decode("utf-8", errors="replace"),
                    node_id=self._compiled.node_id,
                    port_id=self._compiled.output_port_id,
                    interval=context.interval,
                ),
            )
        output = tuple(float(output_values[index]) for index in range(output_count))
        return Emission(
            _reshape_f64(output, self._compiled.output_shape),
            context.interval,
            0,
            (
                EmissionStatus.OK,
                EmissionStatus.DEGRADED,
                EmissionStatus.INVALID,
            )[process_result.status],
            diagnostics,
        )

    def close(self) -> None:
        """native KernelStateを一度だけdestroyする。"""

        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        self._destroy(handle)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


@dataclass(frozen=True)
class _NativeModuleKernel:
    entry: NativeOperationEntry
    parameters: tuple[float, ...]
    implementation_spec: ImplementationSpec
    node_id: int
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    output_port_id: int
    abi_version: str
    process_model: str
    workspace_size_bytes: int
    workspace_alignment_bytes: int
    supports_flush: bool
    session_local: bool
    native_compatible: bool = True

    def create_state(self) -> KernelState[object]:
        """PythonExecutor用のC ABI conformance KernelStateを生成する。"""

        if self.supports_flush:
            raise RuntimeError(
                f"operation={self.entry.operation_id} contract=python_executor_native_flush"
            )
        return _NativeModuleKernelState(self)

    def create_native_runtime_binding(self) -> NativeOperationRuntimeBinding:
        """module handleを保持したC++ runtime bindingを生成する。"""

        values = array("d", self.parameters)
        shape = (len(values),) if values else ()
        return NativeOperationRuntimeBinding(
            self.abi_version,
            self.process_model,
            "float64",
            shape,
            values.tobytes(),
            self.entry.create_address,
            self.entry.process_address,
            self.entry.flush_address,
            self.entry.destroy_address,
            self.entry._module,
        )


def _float64_parameters(operation: OperationSpec, config: ConfigView) -> tuple[float, ...]:
    """ConfigSpec順にscalarまたは数値tupleをC ABI v1 parameter列へ固定する。"""

    values: list[float] = []
    for path, _ in operation.config.fields:
        value = config.require(path)
        if isinstance(value, bool):
            raise TypeError(f"operation={operation.operation_id} config={path} bool is unsupported")
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        if isinstance(value, tuple) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            values.extend(float(item) for item in value)
            continue
        raise TypeError(
            f"operation={operation.operation_id} config={path} contract=native_f64_config_binding"
        )
    return tuple(values)


@dataclass(frozen=True)
class NativeModuleBackend:
    """load済みC ABI moduleからOperation implementationを選択するBackend。

    Args:
        module: process-localにload済みのmodule table。
        name: ImplementationSpecへ記録するBackend名。

    境界条件:
        v1はConfigSpec fieldを宣言順のfloat64 scalar/tuple列としてcreateへ渡す。

    Raises:
        MissingImplementationError: moduleに要求Operationがない場合。
        TypeError: Configをv1 float64 parameter列へ固定できない場合。
    """

    module: NativeOperationModule
    name: str = "cpp"

    def compile_kernel(self, kernel: object, context: CompileContext) -> Kernel[object]:
        """legacy Kernelはmodule Operation境界外として拒否する。

        Args:
            kernel: 対象外のlegacy Kernel。
            context: compile context。

        Raises:
            TypeError: 常に発生し、宣言Operationが必要であることを示す。
        """

        del kernel, context
        raise TypeError("NativeModuleBackend supports declared Operations only")

    def compile_operation(
        self,
        operation: OperationSpec,
        context: object,
    ) -> Kernel[object]:
        """manifest entryと固定Configをcompile済みOperationへbindする。

        Args:
            operation: Backend非依存のOperationSpec。
            context: 不変Configを含むCompileContext。

        Returns:
            PythonExecutorとCppExecutorの両方にbindingできるcompile済みOperation。

        Raises:
            TypeError: contextまたはConfig parameterがv1契約外の場合。
            MissingImplementationError: moduleにoperation IDがない場合。
        """

        if not isinstance(context, CompileContext):
            raise TypeError("NativeModuleBackend requires CompileContext")
        entry = self.module.operations.get(operation.operation_id)
        if entry is None:
            raise MissingImplementationError(
                f"operation={operation.operation_id} backend={self.name} "
                "contract=missing_implementation"
            )
        config = context.config.view(operation.config.scope)
        if context.node_id is None or len(context.output_port_ids) != 1:
            raise TypeError(f"operation={operation.operation_id} contract=native_compile_context")
        if (
            not context.input_shapes
            or any(shape is None for shape in context.input_shapes)
            or len(context.output_shapes) != 1
            or context.output_shapes[0] is None
        ):
            raise TypeError(
                f"node={context.node_id} operation={operation.operation_id} "
                "contract=native_fixed_shape"
            )
        if any(dtype != "float64" for dtype in (*context.input_dtypes, *context.output_dtypes)):
            raise TypeError(
                f"node={context.node_id} operation={operation.operation_id} "
                "contract=native_float64_schema"
            )
        input_shapes = tuple(shape for shape in context.input_shapes if shape is not None)
        output_shape = context.output_shapes[0]
        if output_shape is None:
            raise RuntimeError("validated native output shape was lost")
        specification = entry.implementation_spec(self.name)
        return _NativeModuleKernel(
            entry,
            _float64_parameters(operation, config),
            specification,
            context.node_id,
            input_shapes,
            output_shape,
            context.output_port_ids[0],
            entry.abi_version,
            entry.process_model,
            entry.workspace_size_bytes,
            entry.workspace_alignment_bytes,
            entry.supports_flush,
            entry.session_local,
        )
