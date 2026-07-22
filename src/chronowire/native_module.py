"""process-localなC ABI Operation moduleを読込みBackendへ接続する。"""

from __future__ import annotations

import ctypes
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .config import ConfigView
from .errors import MissingImplementationError
from .kernel import (
    CompileContext,
    CompiledKernel,
    CompiledKernelSession,
    RunContext,
)
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


@dataclass(frozen=True)
class _UnavailableNativeModuleSession:
    operation_id: str

    def run(self, inputs: tuple[object, ...], context: RunContext) -> object:
        del inputs, context
        raise RuntimeError(
            f"operation={self.operation_id} is bound to a C ABI module and requires CppExecutor"
        )


@dataclass(frozen=True)
class _CompiledNativeModuleOperation:
    entry: NativeOperationEntry
    parameters: tuple[float, ...]
    implementation_spec: ImplementationSpec
    abi_version: str
    process_model: str
    workspace_size_bytes: int
    workspace_alignment_bytes: int
    supports_flush: bool
    session_local: bool
    native_compatible: bool = True

    def create_session(self) -> CompiledKernelSession[object]:
        """PythonExecutorでの暗黙fallbackを拒否するsessionを返す。"""

        return _UnavailableNativeModuleSession(self.entry.operation_id)

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

    def compile_kernel(self, kernel: object, context: CompileContext) -> CompiledKernel[object]:
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
    ) -> CompiledKernel[object]:
        """manifest entryと固定Configをcompile済みOperationへbindする。

        Args:
            operation: Backend非依存のOperationSpec。
            context: 不変Configを含むCompileContext。

        Returns:
            CppExecutor用runtime bindingを生成するcompile済みOperation。

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
        specification = entry.implementation_spec(self.name)
        return _CompiledNativeModuleOperation(
            entry,
            _float64_parameters(operation, config),
            specification,
            entry.abi_version,
            entry.process_model,
            entry.workspace_size_bytes,
            entry.workspace_alignment_bytes,
            entry.supports_flush,
            entry.session_local,
        )
