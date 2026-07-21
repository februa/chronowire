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
    """Executorと言語に依存しないserialization可能なExecutionPlan。

    Python callable、collector instance、pointer、allocatorは含めず、安定IDと
    descriptorだけを保持する。schema 0.1/0.2の読込みとround-tripを保証し、
    v0.2ではExecutionBindingsによるprocess-local binding実行を提供する。
    """

    schema_version: str
    kind: str
    backend: str
    nodes: tuple[NodeDescriptor, ...]
    ports: tuple[PortDescriptor, ...]
    edges: tuple[EdgeDescriptor, ...]
    buffers: tuple[BufferDescriptor, ...]
    times: tuple[TimeDescriptor, ...]
    sources: tuple[SourceDescriptor, ...]
    extensions: tuple[ExtensionDescriptor, ...]
    bindings: tuple[BindingDescriptor, ...]
    outputs: tuple[OutputDescriptor, ...]
    diagnostics: tuple[PlanDiagnosticDescriptor, ...]

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
