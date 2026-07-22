"""Python objectを含まないPortablePlanIR descriptorを定義する。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings")
        result[key] = item
    return result


def _items(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be an array")
    return tuple(value)


def _string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_integer(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _boolean(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _integer_tuple(data: Mapping[str, object], key: str) -> tuple[int, ...]:
    values = _items(data.get(key), key)
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} values must be integers")
        result.append(value)
    return tuple(result)


def _optional_integer_tuple(
    data: Mapping[str, object],
    key: str,
) -> tuple[int, ...] | None:
    value = data.get(key)
    if value is None:
        return None
    return _integer_tuple(data, key)


def _string_tuple(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = _items(data.get(key), key)
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{key} values must be strings")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class RationalDescriptor:
    """PortablePlanIRで有理数を正規化して保持する。

    Args:
        numerator: 符号付き分子。
        denominator: 正の分母。

    Raises:
        ValueError: denominatorが正でない場合。
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("rational denominator must be positive")
        normalized = Fraction(self.numerator, self.denominator)
        object.__setattr__(self, "numerator", normalized.numerator)
        object.__setattr__(self, "denominator", normalized.denominator)

    @classmethod
    def from_fraction(cls, value: Fraction) -> RationalDescriptor:
        """Fractionを正規化済みdescriptorへ変換する。"""

        return cls(value.numerator, value.denominator)

    @classmethod
    def from_dict(cls, value: object) -> RationalDescriptor:
        """JSON objectからdescriptorを復元する。

        Raises:
            ValueError: 必須fieldの型または分母が不正な場合。
        """

        data = _mapping(value, "rational descriptor")
        return cls(_integer(data, "numerator"), _integer(data, "denominator"))


@dataclass(frozen=True)
class TimeDescriptor:
    """一つのPortが生成する論理interval列を表す。"""

    time_descriptor_id: int
    timebase: RationalDescriptor
    duration: RationalDescriptor
    period: RationalDescriptor
    offset: RationalDescriptor
    transform: str
    exact: bool
    finite: bool
    generation_end: RationalDescriptor | None

    @classmethod
    def from_dict(cls, value: object) -> TimeDescriptor:
        """JSON objectから時間descriptorを復元する。"""

        data = _mapping(value, "time descriptor")
        offset_value = data.get("offset")
        legacy_value = data.get("phase")
        if offset_value is None and legacy_value is None:
            raise ValueError("time descriptor offset must be present")
        offset = RationalDescriptor.from_dict(
            offset_value if offset_value is not None else legacy_value
        )
        if legacy_value is not None:
            legacy_offset = RationalDescriptor.from_dict(legacy_value)
            if legacy_offset != offset:
                raise ValueError("time descriptor offset conflicts with legacy phase")
        return cls(
            _integer(data, "time_descriptor_id"),
            RationalDescriptor.from_dict(data.get("timebase")),
            RationalDescriptor.from_dict(data.get("duration")),
            RationalDescriptor.from_dict(data.get("period")),
            offset,
            _string(data, "transform"),
            _boolean(data, "exact"),
            _boolean(data, "finite"),
            (
                None
                if data.get("generation_end") is None
                else RationalDescriptor.from_dict(data.get("generation_end"))
            ),
        )

    @property
    def phase(self) -> RationalDescriptor:
        """schema 0.1/0.2利用者向けにoffsetを旧名称で返す。"""

        return self.offset


@dataclass(frozen=True)
class ValueSchemaDescriptor:
    """Port valueのnative表現可否を決める固定schema。"""

    value_schema_id: str
    representation: str
    dtype: str | None
    shape: tuple[int, ...] | None
    strides: tuple[int, ...] | None
    device: str
    read_only: bool

    @classmethod
    def from_dict(cls, value: object) -> ValueSchemaDescriptor:
        """JSON objectからvalue schemaを復元する。"""

        data = _mapping(value, "value schema descriptor")
        return cls(
            _string(data, "value_schema_id"),
            _string(data, "representation"),
            _optional_string(data, "dtype"),
            _optional_integer_tuple(data, "shape"),
            _optional_integer_tuple(data, "strides"),
            _string(data, "device"),
            _boolean(data, "read_only"),
        )


@dataclass(frozen=True)
class NodeDescriptor:
    """ExecutorがPython GraphなしでNodeを識別する固定情報を表す。"""

    node_id: int
    opcode: str
    input_port_ids: tuple[int, ...]
    output_port_ids: tuple[int, ...]
    config_scope_id: str
    execution_domain: str
    binding_slot: str | None
    accepts_invalid: bool
    time_transform_id: int
    max_items: int
    frame_size: int | None = None
    frame_hop: int | None = None
    pad_end: bool = False
    rate_period: RationalDescriptor | None = None
    rate_policy: str | None = None
    callable_time_transform: str = "preserve"
    gap_policy: str = "reset"

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("node max_items must be positive")

    @classmethod
    def from_dict(cls, value: object) -> NodeDescriptor:
        """JSON objectからNode descriptorを復元する。"""

        data = _mapping(value, "node descriptor")
        return cls(
            _integer(data, "node_id"),
            _string(data, "opcode"),
            _integer_tuple(data, "input_port_ids"),
            _integer_tuple(data, "output_port_ids"),
            _string(data, "config_scope_id"),
            _string(data, "execution_domain"),
            _optional_string(data, "binding_slot"),
            _boolean(data, "accepts_invalid"),
            _integer(data, "time_transform_id"),
            _integer(data, "max_items"),
            _optional_integer(data, "frame_size"),
            _optional_integer(data, "frame_hop"),
            _boolean(data, "pad_end") if "pad_end" in data else False,
            (
                None
                if data.get("rate_period") is None
                else RationalDescriptor.from_dict(data.get("rate_period"))
            ),
            _optional_string(data, "rate_policy"),
            _optional_string(data, "callable_time_transform") or "preserve",
            _optional_string(data, "gap_policy") or "reset",
        )


@dataclass(frozen=True)
class PortDescriptor:
    """Node output Portの値・時間・buffer参照を表す。"""

    port_id: int
    producer_node_id: int
    output_index: int
    value_schema_id: str
    time_descriptor_id: int
    sequence_domain: str
    buffer_id: int

    @classmethod
    def from_dict(cls, value: object) -> PortDescriptor:
        """JSON objectからPort descriptorを復元する。"""

        data = _mapping(value, "port descriptor")
        return cls(
            _integer(data, "port_id"),
            _integer(data, "producer_node_id"),
            _integer(data, "output_index"),
            _string(data, "value_schema_id"),
            _integer(data, "time_descriptor_id"),
            _string(data, "sequence_domain"),
            _integer(data, "buffer_id"),
        )


@dataclass(frozen=True)
class EdgeDescriptor:
    """PortBufferからNode inputへ至るconsumer cursorを表す。"""

    edge_id: int
    source_port_id: int
    target_node_id: int
    target_input_index: int
    semantics: str
    keyword: str | None
    buffer_id: int
    cursor_id: int
    required: bool
    adapter_buffer_id: int | None
    tolerance: RationalDescriptor | None = None
    missing_policy: str = "stall"

    @classmethod
    def from_dict(cls, value: object) -> EdgeDescriptor:
        """JSON objectからEdge descriptorを復元する。"""

        data = _mapping(value, "edge descriptor")
        return cls(
            _integer(data, "edge_id"),
            _integer(data, "source_port_id"),
            _integer(data, "target_node_id"),
            _integer(data, "target_input_index"),
            _string(data, "semantics"),
            _optional_string(data, "keyword"),
            _integer(data, "buffer_id"),
            _integer(data, "cursor_id"),
            _boolean(data, "required"),
            _optional_integer(data, "adapter_buffer_id"),
            (
                None
                if data.get("tolerance") is None
                else RationalDescriptor.from_dict(data.get("tolerance"))
            ),
            _optional_string(data, "missing_policy") or "stall",
        )


@dataclass(frozen=True)
class BufferDescriptor:
    """runtime bufferの容量、所有権、解放条件を表す。"""

    buffer_id: int
    kind: str
    producer_port_id: int
    owner_node_id: int | None
    owner_input_index: int | None
    consumer_cursor_ids: tuple[int, ...]
    max_items: int | None
    max_bytes: int | None
    capacity_reasons: tuple[str, ...]
    high_watermark: int
    low_watermark: int
    overflow_policy: str
    reclaim_policy: str
    read_only: bool
    device: str
    alignment_bytes: int | None
    ownership: str
    copy_policy: str

    def __post_init__(self) -> None:
        if self.max_items is not None and self.max_items < 0:
            raise ValueError("buffer max_items must not be negative")
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("buffer max_bytes must not be negative")
        if self.owner_input_index is not None and self.owner_node_id is None:
            raise ValueError("buffer owner input index requires an owner node")
        if self.owner_input_index is not None and self.owner_input_index < 0:
            raise ValueError("buffer owner input index must not be negative")
        if self.high_watermark <= 0:
            raise ValueError("buffer high_watermark must be positive")
        if self.low_watermark < 0 or self.low_watermark >= self.high_watermark:
            raise ValueError("buffer low_watermark must be below high_watermark")
        if self.max_items is not None and self.high_watermark > self.max_items:
            raise ValueError("buffer high_watermark must not exceed max_items")
        if self.alignment_bytes is not None and self.alignment_bytes <= 0:
            raise ValueError("buffer alignment_bytes must be positive")

    @classmethod
    def from_dict(cls, value: object) -> BufferDescriptor:
        """JSON objectからBuffer descriptorを復元する。"""

        data = _mapping(value, "buffer descriptor")
        return cls(
            _integer(data, "buffer_id"),
            _string(data, "kind"),
            _integer(data, "producer_port_id"),
            _optional_integer(data, "owner_node_id"),
            _optional_integer(data, "owner_input_index"),
            _integer_tuple(data, "consumer_cursor_ids"),
            _optional_integer(data, "max_items"),
            _optional_integer(data, "max_bytes"),
            _string_tuple(data, "capacity_reasons"),
            _integer(data, "high_watermark"),
            _integer(data, "low_watermark"),
            _string(data, "overflow_policy"),
            _string(data, "reclaim_policy"),
            _boolean(data, "read_only"),
            _string(data, "device"),
            _optional_integer(data, "alignment_bytes"),
            _string(data, "ownership"),
            _string(data, "copy_policy"),
        )


@dataclass(frozen=True)
class SourceDescriptor:
    """Sourceの制御方式、有限性、request幅を表す。"""

    node_id: int
    mode: str
    is_finite: bool
    request_duration: RationalDescriptor
    burst_max_items: int | None
    ingress_buffer_id: int | None
    overflow_policy: str | None
    gap_policy: str

    @classmethod
    def from_dict(cls, value: object) -> SourceDescriptor:
        """JSON objectからSource descriptorを復元する。"""

        data = _mapping(value, "source descriptor")
        return cls(
            _integer(data, "node_id"),
            _string(data, "mode"),
            _boolean(data, "is_finite"),
            RationalDescriptor.from_dict(data.get("request_duration")),
            _optional_integer(data, "burst_max_items"),
            _optional_integer(data, "ingress_buffer_id"),
            _optional_string(data, "overflow_policy"),
            _string(data, "gap_policy"),
        )


@dataclass(frozen=True)
class BindingDescriptor:
    """PortablePlanIR IDをprocess-local実体へ結ぶslotを表す。"""

    slot_id: str
    kind: str
    node_id: int | None
    port_id: int | None
    abi_version: str

    @classmethod
    def from_dict(cls, value: object) -> BindingDescriptor:
        """JSON objectからBinding descriptorを復元する。"""

        data = _mapping(value, "binding descriptor")
        return cls(
            _string(data, "slot_id"),
            _string(data, "kind"),
            _optional_integer(data, "node_id"),
            _optional_integer(data, "port_id"),
            _string(data, "abi_version"),
        )


@dataclass(frozen=True)
class StageDescriptor:
    """同一runner領域で連続実行できるNode列を表す。

    Args:
        stage_id: Plan内で安定なStage ID。
        node_ids: Stage内でtopological orderに並ぶNode ID。
        execution_domain: compile時に選択した実行領域。
        boundary_reasons: Stage分割理由。
        runner_capabilities: Executorが満たすべきrunner種別。
        boundary_codec: Stage境界で必要な値codec。
        input_port_ids: 他Stageから受け取る入力Port ID。
        output_port_ids: 他StageまたはPlan終端へ渡す出力Port ID。
    """

    stage_id: int
    node_ids: tuple[int, ...]
    execution_domain: str
    boundary_reasons: tuple[str, ...]
    runner_capabilities: tuple[str, ...] = ()
    boundary_codec: str | None = None
    input_port_ids: tuple[int, ...] = ()
    output_port_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ValueError("stage must contain at least one node")

    @classmethod
    def from_dict(cls, value: object) -> StageDescriptor:
        """JSON objectからStage descriptorを復元する。"""

        data = _mapping(value, "stage descriptor")
        return cls(
            _integer(data, "stage_id"),
            _integer_tuple(data, "node_ids"),
            _string(data, "execution_domain"),
            _string_tuple(data, "boundary_reasons"),
            _string_tuple(data, "runner_capabilities") if "runner_capabilities" in data else (),
            _optional_string(data, "boundary_codec"),
            _integer_tuple(data, "input_port_ids") if "input_port_ids" in data else (),
            _integer_tuple(data, "output_port_ids") if "output_port_ids" in data else (),
        )


@dataclass(frozen=True)
class KernelAbiDescriptor:
    """MAP Node bindingが提供する実験的Kernel ABI契約。"""

    node_id: int
    binding_slot: str
    abi_version: str
    process_model: str
    workspace_size_bytes: int | None
    workspace_alignment_bytes: int | None
    supports_flush: bool
    session_local: bool
    native_compatible: bool

    def __post_init__(self) -> None:
        if self.workspace_size_bytes is not None and self.workspace_size_bytes < 0:
            raise ValueError("kernel workspace size must not be negative")
        if self.workspace_alignment_bytes is not None and self.workspace_alignment_bytes <= 0:
            raise ValueError("kernel workspace alignment must be positive")

    @classmethod
    def from_dict(cls, value: object) -> KernelAbiDescriptor:
        """JSON objectからKernel ABI descriptorを復元する。"""

        data = _mapping(value, "kernel ABI descriptor")
        return cls(
            _integer(data, "node_id"),
            _string(data, "binding_slot"),
            _string(data, "abi_version"),
            _string(data, "process_model"),
            _optional_integer(data, "workspace_size_bytes"),
            _optional_integer(data, "workspace_alignment_bytes"),
            _boolean(data, "supports_flush"),
            _boolean(data, "session_local"),
            _boolean(data, "native_compatible"),
        )


@dataclass(frozen=True)
class OperationInputDescriptor:
    """compile後の名前付きOperation入力をPort schemaへ結び付ける。

    Args:
        name: OperationSpec上の入力名。
        port_id: bind済みsource Port。省略optional入力ではNone。
        value_schema_id: resolved input schema。省略optional入力ではNone。
        mode: synchronousまたはlatest。
        required: 実行に必須の入力ならTrue。
        primary: receiver Flowをbindする入力ならTrue。
    """

    name: str
    port_id: int | None
    value_schema_id: str | None
    mode: str
    required: bool
    primary: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation input name must be non-empty")
        if self.mode not in {"synchronous", "latest"}:
            raise ValueError("operation input mode must be synchronous or latest")
        if (self.port_id is None) != (self.value_schema_id is None):
            raise ValueError("operation input port and value schema must both be present or absent")
        if (self.primary or self.required) and self.port_id is None:
            raise ValueError("primary and required operation inputs must be bound")

    @classmethod
    def from_dict(cls, value: object) -> OperationInputDescriptor:
        """JSON objectからOperation input descriptorを復元する。"""

        data = _mapping(value, "operation input descriptor")
        return cls(
            _string(data, "name"),
            _optional_integer(data, "port_id"),
            _optional_string(data, "value_schema_id"),
            _string(data, "mode"),
            _boolean(data, "required"),
            _boolean(data, "primary"),
        )


@dataclass(frozen=True)
class OperationOutputDescriptor:
    """compile後の名前付きOperation出力とEmission規則を表す。

    Args:
        name: OperationSpec上の出力名。
        port_id: 生成先Port ID。
        value_schema_id: resolved output schema ID。
        time_rule: preserveまたはexplicit。
        emission_rule: one、zero_or_one、manyのいずれか。
        max_items: 一回の起動で生成可能な最大Emission件数。

    Raises:
        ValueError: max_itemsが正でない場合。
    """

    name: str
    port_id: int
    value_schema_id: str
    time_rule: str
    emission_rule: str
    max_items: int

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("operation output max_items must be positive")

    @classmethod
    def from_dict(cls, value: object) -> OperationOutputDescriptor:
        """JSON objectからOperation output descriptorを復元する。"""

        data = _mapping(value, "operation output descriptor")
        return cls(
            _string(data, "name"),
            _integer(data, "port_id"),
            _string(data, "value_schema_id"),
            _string(data, "time_rule"),
            _string(data, "emission_rule"),
            _integer(data, "max_items"),
        )


@dataclass(frozen=True)
class ConfigFieldDescriptor:
    """Operationが参照するConfig subtree内の一leaf型依存を表す。

    Args:
        path: subtreeからの相対leaf path。
        type_names: module付き型名。process-local type objectは保持しない。
    """

    path: str
    type_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.path or not self.type_names or any(not item for item in self.type_names):
            raise ValueError("config field path and type names must be non-empty")

    @classmethod
    def from_dict(cls, value: object) -> ConfigFieldDescriptor:
        """JSON objectからConfig field descriptorを復元する。"""

        data = _mapping(value, "config field descriptor")
        return cls(_string(data, "path"), _string_tuple(data, "type_names"))


@dataclass(frozen=True)
class ImplementationDescriptor:
    """選択済みOperation実装のportable ABI属性を表す。

    Python callable、native pointer、module handle、library pathは保持しない。ID、ABI、
    workspace、CPU featureなど別processで再bindを検証できる情報だけを表す。

    Args:
        operation_id: 対象Operation ID。
        implementation_id: 選択済み実装ID。
        backend: 選択Backend名。
        abi_version: 実装ABI version。
        binding_slot: process-local実体の注入slot。
        process_model: scalar/block等の呼出しmodel。
        native_compatible: Python callback不要ならTrue。
        selected_variant: 選択済みCPU variant。
        required_cpu_features: 必須CPU feature。
        workspace_size_bytes: session workspace byte数。
        workspace_alignment_bytes: workspace alignment。
        supports_flush: flush entrypointを持つ場合にTrue。
        session_local: mutable状態がrun-localならTrue。
    """

    operation_id: str
    implementation_id: str
    backend: str
    abi_version: str
    binding_slot: str
    process_model: str
    native_compatible: bool
    selected_variant: str | None
    required_cpu_features: tuple[str, ...]
    workspace_size_bytes: int | None
    workspace_alignment_bytes: int | None
    supports_flush: bool
    session_local: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.operation_id,
                self.implementation_id,
                self.backend,
                self.abi_version,
                self.binding_slot,
                self.process_model,
            )
        ):
            raise ValueError("implementation descriptor identifiers must be non-empty")
        if self.workspace_size_bytes is not None and self.workspace_size_bytes < 0:
            raise ValueError("implementation workspace size must not be negative")
        if self.workspace_alignment_bytes is not None and self.workspace_alignment_bytes <= 0:
            raise ValueError("implementation workspace alignment must be positive")

    @classmethod
    def from_dict(cls, value: object) -> ImplementationDescriptor:
        """JSON objectからImplementation descriptorを復元する。"""

        data = _mapping(value, "implementation descriptor")
        return cls(
            _string(data, "operation_id"),
            _string(data, "implementation_id"),
            _string(data, "backend"),
            _string(data, "abi_version"),
            _string(data, "binding_slot"),
            _string(data, "process_model"),
            _boolean(data, "native_compatible"),
            _optional_string(data, "selected_variant"),
            _string_tuple(data, "required_cpu_features"),
            _optional_integer(data, "workspace_size_bytes"),
            _optional_integer(data, "workspace_alignment_bytes"),
            _boolean(data, "supports_flush"),
            _boolean(data, "session_local"),
        )


@dataclass(frozen=True)
class OperationDescriptor:
    """検証済みOperation意味論と選択Implementation参照を表す。

    Node/Port、resolved schema、Config依存、時間・状態・status規則とbinding slotを固定する。
    compile-time shape resolverやprocess-local実装は責務に含めない。

    Args:
        node_id: 対象MAP Node ID。
        operation_id: 言語非依存Operation ID。
        inputs: 宣言順のresolved入力。
        outputs: 宣言順のresolved出力。
        config_scope_path: Flow Configから選択するsubtree path。
        config_scope_id: bind対象Config scope ID。
        config_digest: compile時Configの完全digest。
        config_fields: subtree内leaf依存。
        state_rule: statelessまたはsession。
        gap_policy: 欠落後の状態規則。
        accepts_invalid: INVALID入力を処理する場合にTrue。
        status_rule: statusとDiagnosticの伝播規則。
        implementation_id: 選択Implementation ID。
        implementation_abi_version: 選択ABI version。
        execution_domain: 選択Backendの実行領域。
        binding_slot: process-local実装の注入slot。
    """

    node_id: int
    operation_id: str
    inputs: tuple[OperationInputDescriptor, ...]
    outputs: tuple[OperationOutputDescriptor, ...]
    config_scope_path: str
    config_scope_id: str
    config_digest: str
    config_fields: tuple[ConfigFieldDescriptor, ...]
    state_rule: str
    gap_policy: str
    accepts_invalid: bool
    status_rule: str
    implementation_id: str
    implementation_abi_version: str
    execution_domain: str
    binding_slot: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.operation_id,
                self.config_scope_id,
                self.config_digest,
                self.status_rule,
                self.implementation_id,
                self.implementation_abi_version,
                self.execution_domain,
                self.binding_slot,
            )
        ):
            raise ValueError("operation descriptor identifiers must be non-empty")
        if sum(item.primary for item in self.inputs) != 1:
            raise ValueError("operation descriptor requires exactly one primary input")
        if not self.outputs:
            raise ValueError("operation descriptor requires at least one output")

    @classmethod
    def from_dict(cls, value: object) -> OperationDescriptor:
        """JSON objectからOperation descriptorを復元する。"""

        data = _mapping(value, "operation descriptor")
        return cls(
            _integer(data, "node_id"),
            _string(data, "operation_id"),
            tuple(
                OperationInputDescriptor.from_dict(item)
                for item in _items(data.get("inputs"), "operation inputs")
            ),
            tuple(
                OperationOutputDescriptor.from_dict(item)
                for item in _items(data.get("outputs"), "operation outputs")
            ),
            _string(data, "config_scope_path"),
            _string(data, "config_scope_id"),
            _string(data, "config_digest"),
            tuple(
                ConfigFieldDescriptor.from_dict(item)
                for item in _items(data.get("config_fields"), "config fields")
            ),
            _string(data, "state_rule"),
            _string(data, "gap_policy"),
            _boolean(data, "accepts_invalid"),
            _string(data, "status_rule"),
            _string(data, "implementation_id"),
            _string(data, "implementation_abi_version"),
            _string(data, "execution_domain"),
            _string(data, "binding_slot"),
        )


@dataclass(frozen=True)
class StreamItemAbiDescriptor:
    """native Stage内で値に付随するEmission情報の固定ABI。"""

    item_abi_id: str
    layout: str
    logical_time_encoding: str
    sequence_encoding: str
    status_encoding: str
    diagnostic_encoding: str
    metadata_encoding: str

    @classmethod
    def from_dict(cls, value: object) -> StreamItemAbiDescriptor:
        """JSON objectからStreamItem ABI descriptorを復元する。"""

        data = _mapping(value, "stream item ABI descriptor")
        return cls(
            _string(data, "item_abi_id"),
            _string(data, "layout"),
            _string(data, "logical_time_encoding"),
            _string(data, "sequence_encoding"),
            _string(data, "status_encoding"),
            _string(data, "diagnostic_encoding"),
            _string(data, "metadata_encoding"),
        )


@dataclass(frozen=True)
class NativeBufferDescriptor:
    """Portの物理native buffer layoutと所有権要求を表す。"""

    port_id: int
    value_schema_id: str
    item_abi_id: str
    layout: str
    alignment_bytes: int
    ownership: str
    read_only: bool

    def __post_init__(self) -> None:
        if self.alignment_bytes <= 0:
            raise ValueError("native buffer alignment must be positive")

    @classmethod
    def from_dict(cls, value: object) -> NativeBufferDescriptor:
        """JSON objectからnative buffer descriptorを復元する。"""

        data = _mapping(value, "native buffer descriptor")
        return cls(
            _integer(data, "port_id"),
            _string(data, "value_schema_id"),
            _string(data, "item_abi_id"),
            _string(data, "layout"),
            _integer(data, "alignment_bytes"),
            _string(data, "ownership"),
            _boolean(data, "read_only"),
        )


@dataclass(frozen=True)
class OutputDescriptor:
    """観測終端とcollectorのportable設定を表す。"""

    index: int
    port_id: int
    collector_kind: str
    max_items: int | None
    overflow_policy: str | None
    binding_slot: str

    @classmethod
    def from_dict(cls, value: object) -> OutputDescriptor:
        """JSON objectからOutput descriptorを復元する。"""

        data = _mapping(value, "output descriptor")
        return cls(
            _integer(data, "index"),
            _integer(data, "port_id"),
            _string(data, "collector_kind"),
            _optional_integer(data, "max_items"),
            _optional_string(data, "overflow_policy"),
            _string(data, "binding_slot"),
        )


@dataclass(frozen=True)
class TriggerDescriptor:
    """Extension発火条件のportableな固定情報を表す。"""

    kind: str
    count: int | None
    period: RationalDescriptor | None
    offset: RationalDescriptor | None

    @classmethod
    def from_dict(cls, value: object) -> TriggerDescriptor:
        """JSON objectからTrigger descriptorを復元する。"""

        data = _mapping(value, "trigger descriptor")
        period = data.get("period")
        offset_value = data.get("offset")
        legacy_value = data.get("phase")
        offset = (
            None
            if offset_value is None and legacy_value is None
            else RationalDescriptor.from_dict(
                offset_value if offset_value is not None else legacy_value
            )
        )
        if legacy_value is not None and offset is not None:
            legacy_offset = RationalDescriptor.from_dict(legacy_value)
            if legacy_offset != offset:
                raise ValueError("trigger descriptor offset conflicts with legacy phase")
        return cls(
            _string(data, "kind"),
            _optional_integer(data, "count"),
            None if period is None else RationalDescriptor.from_dict(period),
            offset,
        )

    @property
    def phase(self) -> RationalDescriptor | None:
        """schema 0.1/0.2利用者向けにoffsetを旧名称で返す。"""

        return self.offset


@dataclass(frozen=True)
class ExtensionDescriptor:
    """compile時に固定したExtension観測契約を表す。"""

    extension_id: str
    observed_port_id: int
    trigger: TriggerDescriptor
    priority: int
    failure_policy: str
    overflow_policy: str
    binding_slot: str
    abi_version: str

    @classmethod
    def from_dict(cls, value: object) -> ExtensionDescriptor:
        """JSON objectからExtension descriptorを復元する。"""

        data = _mapping(value, "extension descriptor")
        return cls(
            _string(data, "extension_id"),
            _integer(data, "observed_port_id"),
            TriggerDescriptor.from_dict(data.get("trigger")),
            _integer(data, "priority"),
            _string(data, "failure_policy"),
            _string(data, "overflow_policy"),
            _string(data, "binding_slot"),
            _string(data, "abi_version"),
        )


@dataclass(frozen=True)
class PlanDiagnosticDescriptor:
    """compile Diagnosticのportableな識別情報を表す。"""

    severity: str
    code: str
    message: str
    node_id: int | None
    port_id: int | None

    @classmethod
    def from_dict(cls, value: object) -> PlanDiagnosticDescriptor:
        """JSON objectからDiagnostic descriptorを復元する。"""

        data = _mapping(value, "diagnostic descriptor")
        return cls(
            _string(data, "severity"),
            _string(data, "code"),
            _string(data, "message"),
            _optional_integer(data, "node_id"),
            _optional_integer(data, "port_id"),
        )


@dataclass(frozen=True)
class PortablePlanIR:
    """Executorと言語に依存しないserialization可能なPlan。

    Python callable、collector instance、pointer、allocatorは含めず、安定IDと
    descriptorだけを保持する。schema 0.1/0.2/0.3/0.4の読込みとround-tripを保証し、
    v0.3ではStage、value schema、実験的Kernel ABI、stream item ABI、
    native buffer layoutを追加する。
    """

    schema_version: str
    kind: str
    backend: str
    nodes: tuple[NodeDescriptor, ...]
    value_schemas: tuple[ValueSchemaDescriptor, ...]
    stages: tuple[StageDescriptor, ...]
    kernel_abis: tuple[KernelAbiDescriptor, ...]
    operations: tuple[OperationDescriptor, ...]
    implementations: tuple[ImplementationDescriptor, ...]
    stream_item_abis: tuple[StreamItemAbiDescriptor, ...]
    native_buffers: tuple[NativeBufferDescriptor, ...]
    ports: tuple[PortDescriptor, ...]
    edges: tuple[EdgeDescriptor, ...]
    buffers: tuple[BufferDescriptor, ...]
    times: tuple[TimeDescriptor, ...]
    sources: tuple[SourceDescriptor, ...]
    extensions: tuple[ExtensionDescriptor, ...]
    bindings: tuple[BindingDescriptor, ...]
    outputs: tuple[OutputDescriptor, ...]
    diagnostics: tuple[PlanDiagnosticDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.operations and not self.implementations:
            return
        if self.schema_version != "0.4":
            raise ValueError("Operation descriptors require PortablePlanIR schema 0.4")
        implementations = {item.binding_slot: item for item in self.implementations}
        bindings = {item.slot_id: item for item in self.bindings}
        nodes = {item.node_id: item for item in self.nodes}
        if len(implementations) != len(self.implementations):
            raise ValueError("implementation binding slots must be unique")
        if len({item.node_id for item in self.operations}) != len(self.operations):
            raise ValueError("operation node IDs must be unique")
        for operation in self.operations:
            implementation = implementations.get(operation.binding_slot)
            binding = bindings.get(operation.binding_slot)
            node = nodes.get(operation.node_id)
            if (
                implementation is None
                or implementation.operation_id != operation.operation_id
                or implementation.implementation_id != operation.implementation_id
                or implementation.abi_version != operation.implementation_abi_version
            ):
                raise ValueError(
                    f"operation node {operation.node_id} has inconsistent implementation"
                )
            if (
                binding is None
                or binding.kind != "operation"
                or binding.node_id != operation.node_id
                or binding.abi_version != operation.implementation_abi_version
            ):
                raise ValueError(f"operation node {operation.node_id} has inconsistent binding")
            if node is None or node.binding_slot != operation.binding_slot:
                raise ValueError(f"operation node {operation.node_id} has inconsistent Node")

    def to_dict(self) -> dict[str, object]:
        """JSON encoderへ渡せるdictへ変換する。"""

        payload = asdict(self)
        # schema 0.1/0.2 readerはphaseを要求するため、移行期間はoffsetと同値で併記する。
        for descriptor in payload["times"]:
            descriptor["phase"] = descriptor["offset"]
        for extension in payload["extensions"]:
            trigger = extension["trigger"]
            trigger["phase"] = trigger["offset"]
        return payload

    def to_json(self) -> str:
        """UTF-8保存用の整形済みJSON文字列を返す。"""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: object) -> PortablePlanIR:
        """検証済みJSON objectからPortablePlanIRを復元する。

        Raises:
            ValueError: schemaまたはdescriptor fieldの型が不正な場合。
        """

        data = _mapping(value, "portable plan")
        return cls(
            schema_version=_string(data, "schema_version"),
            kind=_string(data, "kind"),
            backend=_string(data, "backend"),
            nodes=tuple(
                NodeDescriptor.from_dict(item) for item in _items(data.get("nodes"), "nodes")
            ),
            value_schemas=tuple(
                ValueSchemaDescriptor.from_dict(item)
                for item in _items(data.get("value_schemas", ()), "value_schemas")
            ),
            stages=tuple(
                StageDescriptor.from_dict(item) for item in _items(data.get("stages", ()), "stages")
            ),
            kernel_abis=tuple(
                KernelAbiDescriptor.from_dict(item)
                for item in _items(data.get("kernel_abis", ()), "kernel_abis")
            ),
            operations=tuple(
                OperationDescriptor.from_dict(item)
                for item in _items(data.get("operations", ()), "operations")
            ),
            implementations=tuple(
                ImplementationDescriptor.from_dict(item)
                for item in _items(data.get("implementations", ()), "implementations")
            ),
            stream_item_abis=tuple(
                StreamItemAbiDescriptor.from_dict(item)
                for item in _items(data.get("stream_item_abis", ()), "stream_item_abis")
            ),
            native_buffers=tuple(
                NativeBufferDescriptor.from_dict(item)
                for item in _items(data.get("native_buffers", ()), "native_buffers")
            ),
            ports=tuple(
                PortDescriptor.from_dict(item) for item in _items(data.get("ports"), "ports")
            ),
            edges=tuple(
                EdgeDescriptor.from_dict(item) for item in _items(data.get("edges"), "edges")
            ),
            buffers=tuple(
                BufferDescriptor.from_dict(item) for item in _items(data.get("buffers"), "buffers")
            ),
            times=tuple(
                TimeDescriptor.from_dict(item) for item in _items(data.get("times"), "times")
            ),
            sources=tuple(
                SourceDescriptor.from_dict(item) for item in _items(data.get("sources"), "sources")
            ),
            extensions=tuple(
                ExtensionDescriptor.from_dict(item)
                for item in _items(data.get("extensions"), "extensions")
            ),
            bindings=tuple(
                BindingDescriptor.from_dict(item)
                for item in _items(data.get("bindings"), "bindings")
            ),
            outputs=tuple(
                OutputDescriptor.from_dict(item) for item in _items(data.get("outputs"), "outputs")
            ),
            diagnostics=tuple(
                PlanDiagnosticDescriptor.from_dict(item)
                for item in _items(data.get("diagnostics"), "diagnostics")
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> PortablePlanIR:
        """JSON文字列をparseしてPortablePlanIRを復元する。

        Raises:
            ValueError: JSONまたはschemaが不正な場合。
        """

        try:
            value: object = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid PortablePlanIR JSON: {error}") from error
        return cls.from_dict(value)
