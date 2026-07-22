"""Flow境界の言語非依存Operation宣言を定義する。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

from .config import ConfigView
from .kernel import CompiledKernel, GapPolicy, RunContext

Dimension: TypeAlias = int | str | None
ShapeResolver: TypeAlias = Callable[
    [Mapping[str, object], ConfigView],
    tuple[int, ...],
]

_INPUT_MODES = frozenset({"synchronous", "latest"})
_STATES = frozenset({"stateless", "session"})
_TIME_RULES = frozenset({"preserve", "explicit"})
_EMISSION_RULES = frozenset({"one", "zero_or_one", "many"})


@dataclass(frozen=True)
class ValueSpec:
    """Emission一件の値schemaを宣言する。

    Args:
        dtype: 要素型。NoneはPython opaque値など未指定の型。
        shape: 固定値、symbol、`$config.`参照、None dimensionからなるitem shape。
        device: 値を置くdevice名。
        representation: 物理表現。Noneではdtypeから既定値を決める。
        read_only: Operation入力から値を書き換えない契約。

    Raises:
        ValueError: dtype、shape、device、representationが不正な場合。

    境界条件:
        shapeは物理batch件数を含まず、一Emissionの値だけを表す。
    """

    dtype: str | None = None
    shape: tuple[Dimension, ...] | None = None
    device: str = "cpu"
    representation: str | None = None
    read_only: bool = True

    def __post_init__(self) -> None:
        if self.dtype is not None and not self.dtype:
            raise ValueError("ValueSpec dtype must be non-empty or None")
        if not self.device:
            raise ValueError("ValueSpec device must be non-empty")
        if self.representation is not None and not self.representation:
            raise ValueError("ValueSpec representation must be non-empty or None")
        if self.shape is None:
            return
        for dimension in self.shape:
            if isinstance(dimension, bool):
                raise ValueError("ValueSpec dimensions must not be bool")
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("ValueSpec fixed dimensions must be positive")
            if isinstance(dimension, str):
                invalid_config_reference = dimension.startswith("$") and not dimension.startswith(
                    "$config."
                )
                if not dimension or dimension == "$config." or invalid_config_reference:
                    raise ValueError("ValueSpec symbolic dimension is invalid")
            elif dimension is not None and not isinstance(dimension, int):
                raise ValueError("ValueSpec dimensions must be int, str, or None")

    @property
    def resolved_representation(self) -> str:
        """明示値またはdtypeから決めた既定representationを返す。"""

        if self.representation is not None:
            return self.representation
        if self.dtype is None:
            return "python_opaque"
        if self.dtype == "float64":
            return "contiguous_f64"
        return f"contiguous_{self.dtype}"


@dataclass(frozen=True)
class OperationInputSpec:
    """一つの名前付きOperation入力を宣言する。

    Args:
        value: 入力一件のValueSpec。
        primary: receiver Flowをbindする入力ならTrue。
        mode: `synchronous`またはlatest StateFlow用の`latest`。
        required: 入力を省略できない場合にTrue。
    """

    value: ValueSpec = ValueSpec()
    primary: bool = False
    mode: str = "synchronous"
    required: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _INPUT_MODES:
            raise ValueError("Operation input mode must be synchronous or latest")
        if self.primary and self.mode != "synchronous":
            raise ValueError("Operation primary input must be synchronous")


@dataclass(frozen=True)
class OperationOutputSpec:
    """一つのOperation output Portを宣言する。

    Args:
        value: 出力ValueSpec、またはprimary入力schemaを保つ`same`。
        time: intervalを保持する`preserve`または明示変更する`explicit`。
        emissions: `one`、`zero_or_one`、`many`のいずれか。
        max_items: 一回の起動で生成できる最大Emission件数。
    """

    value: ValueSpec | str = "same"
    time: str = "preserve"
    emissions: str = "one"
    max_items: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.value, str) and self.value != "same":
            raise ValueError("Operation output string value must be 'same'")
        if self.time not in _TIME_RULES:
            raise ValueError("Operation output time must be preserve or explicit")
        if self.emissions not in _EMISSION_RULES:
            raise ValueError("Operation output emissions rule is invalid")
        if self.max_items <= 0:
            raise ValueError("Operation output max_items must be positive")
        if self.emissions != "many" and self.max_items != 1:
            raise ValueError("only many-emission outputs may set max_items above one")


@dataclass(frozen=True, init=False)
class ConfigSpec:
    """Operationへ渡すConfig subtreeと必須leaf型を宣言する。

    Args:
        scope: FlowのConfigから選択するdot区切りsubtree。空文字はroot。
        fields: subtreeからの相対leaf pathと期待型。

    Raises:
        ValueError: scope、path、型宣言が不正な場合。
    """

    scope: str
    fields: tuple[tuple[str, type[object] | tuple[type[object], ...]], ...]

    def __init__(
        self,
        scope: str = "",
        fields: Mapping[str, type[object] | tuple[type[object], ...]] | None = None,
    ) -> None:
        if scope.startswith(".") or scope.endswith(".") or ".." in scope:
            raise ValueError("ConfigSpec scope must be a normalized dot path")
        normalized: list[tuple[str, type[object] | tuple[type[object], ...]]] = []
        for path, expected in (fields or {}).items():
            if not path or path.startswith(".") or path.endswith(".") or ".." in path:
                raise ValueError("ConfigSpec field must be a normalized relative path")
            expected_types = expected if isinstance(expected, tuple) else (expected,)
            if not expected_types or any(not isinstance(item, type) for item in expected_types):
                raise ValueError("ConfigSpec field types must be type objects")
            normalized.append((path, expected))
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "fields", tuple(sorted(normalized)))


@dataclass(frozen=True)
class OperationSpec:
    """compile前の言語非依存Operation意味論を表す。

    Args:
        operation_id: DSP packageが安定して所有する一意なID。
        inputs: 名前と入力宣言の順序付き集合。
        outputs: 名前と出力宣言の順序付き集合。
        config: Operationへ渡す固定Config subtree契約。
        state: `stateless`またはrun-local状態を要求する`session`。
        gap_policy: 入力欠落後のsession状態規則。
        accepts_invalid: INVALID入力でも実装を呼び出す場合にTrue。
        shape_resolver: compile時だけ呼ぶ出力shape解決関数。

    Raises:
        ValueError: ID、入出力名、primary数、状態規則が不正な場合。

    境界条件:
        Python callableやnative pointerは保持せず、実装情報とrun-local状態を分離する。
    """

    operation_id: str
    inputs: tuple[tuple[str, OperationInputSpec], ...]
    outputs: tuple[tuple[str, OperationOutputSpec], ...]
    config: ConfigSpec = ConfigSpec()
    state: str = "stateless"
    gap_policy: GapPolicy = GapPolicy.RESET
    accepts_invalid: bool = False
    shape_resolver: ShapeResolver | None = None

    def __post_init__(self) -> None:
        if not self.operation_id or any(part == "" for part in self.operation_id.split(".")):
            raise ValueError("OperationSpec operation_id must be a stable dotted ID")
        input_names = tuple(name for name, _ in self.inputs)
        output_names = tuple(name for name, _ in self.outputs)
        if not input_names or len(set(input_names)) != len(input_names):
            raise ValueError("OperationSpec input names must be non-empty and unique")
        if not output_names or len(set(output_names)) != len(output_names):
            raise ValueError("OperationSpec output names must be non-empty and unique")
        primary_count = sum(item.primary for _, item in self.inputs)
        if primary_count != 1:
            raise ValueError("OperationSpec requires exactly one primary input")
        if self.state not in _STATES:
            raise ValueError("OperationSpec state must be stateless or session")
        if not isinstance(self.gap_policy, GapPolicy):
            raise ValueError("OperationSpec gap_policy must be a GapPolicy")

    @property
    def primary_input_name(self) -> str:
        """receiver Flowをbindする一意なprimary入力名を返す。"""

        return next(name for name, item in self.inputs if item.primary)

    @property
    def input_mapping(self) -> Mapping[str, OperationInputSpec]:
        """入力宣言の読み取り専用mappingを返す。"""

        return MappingProxyType(dict(self.inputs))


@dataclass(frozen=True)
class ImplementationSpec:
    """operation IDに対応する一つの実装候補のportable属性を表す。

    Args:
        operation_id: 対象OperationSpecの安定ID。
        implementation_id: 実装候補を一意に識別するID。
        backend: 実装を選択するBackend名。
        abi_version: process境界で照合するABI version。
        native_compatible: Python callbackなしでnative実行できる場合にTrue。

    Raises:
        ValueError: 必須ID、Backend名、ABI versionが空の場合。
    """

    operation_id: str
    implementation_id: str
    backend: str
    abi_version: str
    native_compatible: bool = False

    def __post_init__(self) -> None:
        if not all((self.operation_id, self.implementation_id, self.backend, self.abi_version)):
            raise ValueError("ImplementationSpec IDs, backend, and ABI must be non-empty")


@dataclass(frozen=True)
class ImplementationBinding:
    """process-local実装実体とportable ImplementationSpecを結び付ける。

    Args:
        spec: Planへ記録可能な実装属性。
        implementation: 現processだけで参照する実装実体。

    境界条件:
        callableはPortablePlanIRへ直列化せず、run-local状態も保持しない。
    """

    spec: ImplementationSpec
    implementation: Callable[..., object]


@runtime_checkable
class OperationBackend(Protocol):
    """OperationSpecを選択Backend実装へcompileする追加protocol。

    Backendは実装選択とcompileだけを行い、SchedulerやEdge bufferを運用しない。
    """

    @property
    def name(self) -> str:
        """Backend名を返す。"""

        ...

    def compile_operation(
        self,
        operation: OperationSpec,
        context: object,
    ) -> CompiledKernel[object]:
        """operation IDに対応する実装を選択してcompileする。

        Args:
            operation: Backendに依存しない検証済み意味論。
            context: 固定Configとcompile定数を提供するcontext。

        Returns:
            runごとに独立sessionを生成できる内部factory。

        Raises:
            Exception: 対応実装不足またはABI不整合を検出した場合。
        """

        ...


@dataclass(frozen=True)
class OperationDefinition:
    """OperationSpecと任意のPython参照実装をFlow.mapへ渡すhandle。

    Args:
        spec: 言語非依存のOperation宣言。
        python_binding: Python参照実装。C++ first宣言ではNone。

    境界条件:
        Flow構築時は実装を呼ばず、compile時のBackend選択まで保持する。
    """

    spec: OperationSpec
    python_binding: ImplementationBinding | None = None

    @property
    def operation_id(self) -> str:
        """安定operation IDを返す。"""

        return self.spec.operation_id


@dataclass(frozen=True)
class _PythonOperationSession:
    """名前付き入力とConfigViewをPython参照実装へ渡すrun-local session。"""

    implementation: Callable[..., object]
    input_names: tuple[str, ...]
    config: ConfigView

    def run(self, inputs: tuple[object, ...], context: RunContext) -> object:
        """物理tupleを不変な名前付き論理入力集合へ変換して実行する。"""

        del context
        if len(inputs) != len(self.input_names):
            raise ValueError("Operation input count does not match its compiled declaration")
        named_inputs = MappingProxyType(dict(zip(self.input_names, inputs, strict=True)))
        return self.implementation(named_inputs, self.config)


@dataclass(frozen=True)
class _CompiledPythonOperation:
    """Python OperationSessionを生成するcompile済みfactory。"""

    implementation: Callable[..., object]
    input_names: tuple[str, ...]
    config: ConfigView

    def create_session(self) -> _PythonOperationSession:
        """別runと状態を共有しないPython OperationSessionを返す。"""

        return _PythonOperationSession(self.implementation, self.input_names, self.config)


def compile_python_operation(
    definition: OperationDefinition,
    *,
    input_names: tuple[str, ...],
    config: ConfigView,
) -> CompiledKernel[object]:
    """OperationDefinitionのPython参照実装を内部factoryへcompileする。

    Raises:
        ValueError: Python実装が存在しない場合。
    """

    if definition.python_binding is None:
        raise ValueError(f"operation {definition.operation_id!r} has no Python implementation")
    return _CompiledPythonOperation(
        definition.python_binding.implementation,
        input_names,
        config,
    )


def _normalize_inputs(
    inputs: Mapping[str, OperationInputSpec] | None,
) -> tuple[tuple[str, OperationInputSpec], ...]:
    if inputs is None:
        return (("input", OperationInputSpec(primary=True)),)
    return tuple(inputs.items())


def _normalize_outputs(
    output: str | OperationOutputSpec | Mapping[str, OperationOutputSpec],
) -> tuple[tuple[str, OperationOutputSpec], ...]:
    if isinstance(output, str):
        if output != "same":
            raise ValueError("operation output shorthand must be 'same'")
        return (("output", OperationOutputSpec()),)
    if isinstance(output, OperationOutputSpec):
        return (("output", output),)
    return tuple(output.items())


def _make_spec(
    *,
    operation_id: str,
    inputs: Mapping[str, OperationInputSpec] | None,
    output: str | OperationOutputSpec | Mapping[str, OperationOutputSpec],
    config: ConfigSpec | None,
    config_scope: str | None,
    state: str,
    gap: GapPolicy | str,
    accepts_invalid: bool,
    shape_resolver: ShapeResolver | None,
) -> OperationSpec:
    if config is not None and config_scope is not None:
        raise ValueError("operation accepts either config or config_scope, not both")
    resolved_config = config if config is not None else ConfigSpec(config_scope or "")
    resolved_gap = GapPolicy(gap)
    return OperationSpec(
        operation_id,
        _normalize_inputs(inputs),
        _normalize_outputs(output),
        resolved_config,
        state,
        resolved_gap,
        accepts_invalid,
        shape_resolver,
    )


def operation(
    *,
    operation_id: str,
    inputs: Mapping[str, OperationInputSpec] | None = None,
    output: str | OperationOutputSpec | Mapping[str, OperationOutputSpec] = "same",
    config: ConfigSpec | None = None,
    config_scope: str | None = None,
    state: str = "stateless",
    gap: GapPolicy | str = GapPolicy.RESET,
    accepts_invalid: bool = False,
    shape_resolver: ShapeResolver | None = None,
) -> Callable[[Callable[..., object]], OperationDefinition]:
    """Python参照実装を持つOperationDefinitionを生成するdecorator。

    Args:
        operation_id: DSP packageが所有する安定ID。
        inputs: 名前付き入力宣言。Noneでは`input`をprimaryにする。
        output: 単一出力または名前付き複数出力宣言。
        config: 固定Config subtreeとleaf型の契約。
        config_scope: leaf型を宣言しない簡易Config scope。
        state: `stateless`または`session`。
        gap: 入力欠落後の状態規則。
        accepts_invalid: INVALID入力を実装へ渡す場合にTrue。
        shape_resolver: compile時だけ使う単一出力shape resolver。

    Returns:
        `func(inputs, config)`をOperationDefinitionへ変換するdecorator。

    Raises:
        ValueError: 宣言が不正、またはconfigとconfig_scopeを併用した場合。

    境界条件:
        decoratorは関数を実行せず、Python実装をPortablePlanIRへ保存しない。
    """

    spec = _make_spec(
        operation_id=operation_id,
        inputs=inputs,
        output=output,
        config=config,
        config_scope=config_scope,
        state=state,
        gap=gap,
        accepts_invalid=accepts_invalid,
        shape_resolver=shape_resolver,
    )

    def decorate(implementation: Callable[..., object]) -> OperationDefinition:
        binding = ImplementationBinding(
            ImplementationSpec(
                operation_id,
                f"{operation_id}.python",
                "python",
                "chronowire.operation.python.v1",
            ),
            implementation,
        )
        return OperationDefinition(spec, binding)

    return decorate


def declare_operation(
    *,
    operation_id: str,
    inputs: Mapping[str, OperationInputSpec] | None = None,
    output: str | OperationOutputSpec | Mapping[str, OperationOutputSpec] = "same",
    config: ConfigSpec | None = None,
    config_scope: str | None = None,
    state: str = "stateless",
    gap: GapPolicy | str = GapPolicy.RESET,
    accepts_invalid: bool = False,
    shape_resolver: ShapeResolver | None = None,
) -> OperationDefinition:
    """Python実装を持たない言語非依存Operation宣言を生成する。

    Args:
        operation_id: DSP packageが所有する安定ID。
        inputs: 名前付き入力宣言。Noneでは`input`をprimaryにする。
        output: 単一出力または名前付き複数出力宣言。
        config: 固定Config subtreeとleaf型の契約。
        config_scope: leaf型を宣言しない簡易Config scope。
        state: `stateless`または`session`。
        gap: 入力欠落後の状態規則。
        accepts_invalid: INVALID入力を実装へ渡す場合にTrue。
        shape_resolver: compile時だけ使う単一出力shape resolver。

    Returns:
        native Backendがoperation IDで実装を選択できるOperationDefinition。

    Raises:
        ValueError: 宣言が不正、またはconfigとconfig_scopeを併用した場合。

    境界条件:
        Python Backendで選択するとMissingImplementationErrorになる。
    """

    return OperationDefinition(
        _make_spec(
            operation_id=operation_id,
            inputs=inputs,
            output=output,
            config=config,
            config_scope=config_scope,
            state=state,
            gap=gap,
            accepts_invalid=accepts_invalid,
            shape_resolver=shape_resolver,
        )
    )
