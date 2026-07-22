"""Logical Graphのcompileと単一thread決定的runtimeを実装する。"""

from __future__ import annotations

import inspect
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, TypeVar

from .collector import Bounded, Collector, CollectorSession, Latest, NoCollect, Sink
from .errors import (
    CompileError,
    DuplicateExtensionIdError,
    DuplicateOutputError,
    ExtensionBindingError,
    ExtensionExecutionError,
    KernelExecutionError,
    MissingConfigError,
    MissingImplementationError,
    PlanSessionError,
    ShapeMismatchError,
    SourceExecutionError,
)
from .executor import (
    CppExecutor,
    CythonExecutor,
    Executor,
    ExecutorPlanSession,
    ExecutorSession,
    PythonExecutor,
)
from .extension import (
    Always,
    Every,
    EveryLogicalTime,
    Extension,
    ExtensionSession,
    ObservationSpec,
    OutputEvent,
    PlanContext,
    TriggerSession,
)
from .graph import (
    Flow,
    Graph,
    InputSemantics,
    InputSpec,
    MissingInputPolicy,
    NodeKind,
    NodeSpec,
    RatePolicy,
)
from .kernel import (
    Backend,
    CompileContext,
    CompiledKernel,
    CompiledKernelSession,
    GapPolicy,
    Kernel,
    NativeCompiledKernel,
    NativeValueSchemaProvider,
    PythonBackend,
    PythonCallableKernel,
    RunContext,
)
from .model import (
    Diagnostic,
    Emission,
    EmissionStatus,
    EmitMany,
    KernelOutputs,
    LogicalInterval,
    LogicalTime,
    Severity,
    Skip,
)
from .native import F64SourceValues, F64VectorSourceValues, IdentityF64Kernel
from .operation import (
    CompiledOperationMetadata,
    OperationBackend,
    OperationDefinition,
    ValueSpec,
    compile_python_operation,
)
from .plan_ir import (
    BindingDescriptor,
    BufferDescriptor,
    ConfigFieldDescriptor,
    EdgeDescriptor,
    ExtensionDescriptor,
    ImplementationDescriptor,
    KernelAbiDescriptor,
    NativeBufferDescriptor,
    NodeDescriptor,
    OperationDescriptor,
    OperationInputDescriptor,
    OperationOutputDescriptor,
    OutputDescriptor,
    PlanDiagnosticDescriptor,
    PortablePlanIR,
    PortDescriptor,
    RationalDescriptor,
    SourceDescriptor,
    StageDescriptor,
    StreamItemAbiDescriptor,
    TimeDescriptor,
    TriggerDescriptor,
    ValueSchemaDescriptor,
)
from .runtime_buffer import (
    CursorQueue,
    FrameHistoryBuffer,
    GapMarker,
    LatestStateBuffer,
    PortBuffer,
    RealtimeIngressBuffer,
)
from .source import (
    RealtimeOverflowPolicy,
    RealtimeSource,
    RealtimeSourceSession,
    Source,
    SourceRequest,
)

T = TypeVar("T")
_SOURCE_BOUNDARY = object()


@dataclass(frozen=True)
class OutputSpec(Generic[T]):
    """観測終端Flowとcollector policyの組を表す。"""

    flow: Flow[T]
    collector: Collector[T]


def output(flow: Flow[T], *, collector: Collector[T]) -> OutputSpec[T]:
    """FlowとcollectorからOutputSpecを作成する。"""

    return OutputSpec(flow, collector)


@dataclass(frozen=True)
class OutputResult(Generic[T]):
    """一つのcompile outputについてcollectorが得たrun結果を表す。"""

    emissions: tuple[Emission[T], ...]
    collector_kind: str
    received_count: int
    dropped_count: int
    logical_start: LogicalTime | None
    logical_end: LogicalTime | None


@dataclass(frozen=True)
class RunResult:
    """一回のExecutionPlan.runの結果と診断summaryを表す。"""

    outputs: tuple[OutputResult[Any], ...]
    diagnostics: tuple[Diagnostic, ...]
    status_counts: dict[EmissionStatus, int]
    completed: bool
    profile: SessionProfile | None = None


@dataclass(frozen=True)
class KernelProfile:
    """一つのMAP Nodeのsession内実行時間summary。"""

    node_id: int
    call_count: int
    total_ns: int
    max_ns: int


@dataclass(frozen=True)
class BufferProfile:
    """一つのruntime bufferの使用量summary。"""

    buffer_id: int
    kind: str
    capacity: int
    current_items: int
    high_watermark: int


@dataclass(frozen=True)
class SourceProfile:
    """一つのSourceの配送、pending、drop summary。"""

    node_id: int
    emitted_count: int
    pending_items: int
    dropped_count: int
    logical_end: Fraction | None


@dataclass(frozen=True)
class SessionProfile:
    """Profiler有効時だけRunResultへ付加するrun-local snapshot。"""

    scheduler_steps: int
    kernels: tuple[KernelProfile, ...]
    buffers: tuple[BufferProfile, ...]
    sources: tuple[SourceProfile, ...]


@dataclass(frozen=True)
class RuntimeOptions:
    """Executorのchunk、buffer watermark、実行budgetを指定する。

    Args:
        source_chunk_duration: pull Source一回の要求幅。Noneはcompile済み既定値。
        port_high_watermark: 全PORT_SHARED bufferの最小capacity。Noneはcompile済み値。
        port_low_watermark: pull再開目安。high未満の非負整数。
        max_scheduler_steps: 一回のrunまたはrun_untilで進める実行単位上限。
        profiler_enabled: session profilerを有効化する場合にTrue。

    Raises:
        ValueError: 値が正でない、またはwatermark関係が不正な場合。
    """

    source_chunk_duration: Fraction | None = None
    port_high_watermark: int | None = None
    port_low_watermark: int | None = None
    max_scheduler_steps: int | None = None
    profiler_enabled: bool = False

    def __post_init__(self) -> None:
        if self.source_chunk_duration is not None and self.source_chunk_duration <= 0:
            raise ValueError("source_chunk_duration must be positive")
        if self.port_high_watermark is not None and self.port_high_watermark <= 0:
            raise ValueError("port_high_watermark must be positive")
        if self.port_low_watermark is not None and self.port_low_watermark < 0:
            raise ValueError("port_low_watermark must not be negative")
        if self.port_low_watermark is not None and self.port_high_watermark is None:
            raise ValueError("port_low_watermark requires port_high_watermark")
        if (
            self.port_low_watermark is not None
            and self.port_high_watermark is not None
            and self.port_low_watermark >= self.port_high_watermark
        ):
            raise ValueError("port_low_watermark must be below port_high_watermark")
        if self.max_scheduler_steps is not None and self.max_scheduler_steps <= 0:
            raise ValueError("max_scheduler_steps must be positive")


@dataclass
class _FrameState:
    history: FrameHistoryBuffer[Emission[object]]
    skip_remaining: int = 0


@dataclass
class _RateState:
    """一つのRATE Nodeについて次の発火時刻を保持するrun-local状態。"""

    next_fire: Fraction | None = None


@dataclass(frozen=True)
class _TimingProof:
    """compile時のrate/frame境界証明に必要な最小情報を保持する。"""

    exact: bool
    contains_frame: bool
    contains_rate: bool


def _status_rank(status: EmissionStatus) -> int:
    return {
        EmissionStatus.OK: 0,
        EmissionStatus.DEGRADED: 1,
        EmissionStatus.INVALID: 2,
    }[status]


def _combined_status(emissions: Sequence[Emission[object]]) -> EmissionStatus:
    return max((item.status for item in emissions), key=_status_rank, default=EmissionStatus.OK)


def _has_input_overrun(emission: Emission[object]) -> bool:
    """Emissionがrealtime欠落境界直後の値ならTrueを返す。"""

    return any(item.code == "INPUT_OVERRUN" for item in emission.diagnostics)


def _required_nodes(graph: Graph, root_ports: Sequence[int]) -> tuple[NodeSpec, ...]:
    required: set[int] = set()

    def visit_port(port_id: int) -> None:
        node = graph.node_for_port(port_id)
        if node.id in required:
            return
        required.add(node.id)
        for item in node.inputs:
            visit_port(item.source_port)

    for port_id in root_ports:
        visit_port(port_id)
    return tuple(node for node in graph.nodes if node.id in required)


def _time_signature(nodes: Sequence[NodeSpec]) -> dict[int, tuple[Fraction, Fraction]]:
    signatures: dict[int, tuple[Fraction, Fraction]] = {}
    for node in nodes:
        if node.kind is NodeKind.SOURCE:
            value = (Fraction(1), Fraction(1))
        elif node.kind is NodeKind.MAP:
            value = signatures[node.inputs[0].source_port]
        elif node.kind is NodeKind.FRAME:
            input_length, input_step = signatures[node.inputs[0].source_port]
            if node.frame_size is None or node.frame_hop is None:
                raise RuntimeError("FRAME Node lacks size or hop")
            length = input_length + (node.frame_size - 1) * input_step
            value = (length, node.frame_hop * input_step)
        elif node.kind is NodeKind.RATE:
            if node.rate_period is None:
                raise RuntimeError("RATE Node lacks period")
            if node.rate_policy is RatePolicy.SAMPLE:
                input_length, _ = signatures[node.inputs[0].source_port]
                value = (input_length, node.rate_period)
            else:
                value = (node.rate_period, node.rate_period)
        else:
            raise RuntimeError(f"unsupported Node kind {node.kind!r}")
        for output_port in node.output_ports:
            signatures[output_port] = value
    return signatures


def _validate_rate_frame_boundaries(nodes: Sequence[NodeSpec]) -> None:
    """RATEとFRAMEの順序および同期格子を静的に検証する。

    RATEはitem列の論理格子を確定する境界であり、FRAMEより前に置く。完成済み
    frameへRATEを適用するとHOLDによる重複または未使用frameが生じ得るため拒否する。
    また、明示time transform後の未知格子をFRAMEへ渡す場合と、RATEを含む同期入力の
    格子不一致も、runtimeへ先送りせずcompile違反とする。
    """

    signatures = _time_signature(nodes)
    proofs: dict[int, _TimingProof] = {}
    for node in nodes:
        if node.kind is NodeKind.SOURCE:
            proof = _TimingProof(exact=True, contains_frame=False, contains_rate=False)
        elif node.kind is NodeKind.MAP:
            main = proofs[node.inputs[0].source_port]
            synchronous = [
                item for item in node.inputs if item.semantics is InputSemantics.SYNCHRONOUS
            ]
            synchronous_proofs = [proofs[item.source_port] for item in synchronous]
            rate_sensitive = any(item.contains_rate for item in synchronous_proofs)
            if rate_sensitive and any(not item.exact for item in synchronous_proofs):
                ports = tuple(item.source_port for item in synchronous)
                raise CompileError(
                    f"node {node.id} port {node.output_port} cannot prove synchronous "
                    f"rate/frame boundaries for input ports {ports}; insert an explicit "
                    "Flow.rate(...) after the time-transforming Kernel and before Flow.frame(...); "
                    "contract=stable_rate_frame_boundary"
                )
            input_signatures = {signatures[item.source_port] for item in synchronous}
            if rate_sensitive and len(input_signatures) > 1:
                ports = tuple(item.source_port for item in synchronous)
                raise CompileError(
                    f"node {node.id} port {node.output_port} has incompatible synchronous "
                    f"rate/frame grids on input ports {ports}: "
                    f"{tuple(str(value) for value in sorted(input_signatures))}; insert explicit "
                    "Flow.rate(...).frame(...) stages that produce identical duration and period; "
                    "contract=stable_rate_frame_boundary"
                )
            if node.time_transform == "explicit":
                # 外部resampling Kernelは旧格子を終了する。ただし新格子は未知なので、
                # FRAMEへ進むには直後のRATEでperiodを再宣言しなければならない。
                proof = _TimingProof(
                    exact=False,
                    contains_frame=False,
                    contains_rate=False,
                )
            else:
                proof = _TimingProof(
                    exact=main.exact,
                    contains_frame=main.contains_frame,
                    contains_rate=main.contains_rate,
                )
        elif node.kind is NodeKind.FRAME:
            source_port = node.inputs[0].source_port
            source = proofs[source_port]
            if not source.exact:
                raise CompileError(
                    f"frame node {node.id} port {node.output_port} cannot prove the input grid "
                    f"from port {source_port}; insert explicit Flow.rate(...) immediately before "
                    "Flow.frame(...); contract=stable_rate_frame_boundary"
                )
            proof = _TimingProof(
                exact=True,
                contains_frame=True,
                contains_rate=source.contains_rate,
            )
        elif node.kind is NodeKind.RATE:
            source_port = node.inputs[0].source_port
            source = proofs[source_port]
            if source.contains_frame and node.rate_policy is RatePolicy.HOLD:
                raise CompileError(
                    f"rate node {node.id} port {node.output_port} consumes a completed frame "
                    f"from port {source_port}; move Flow.rate(...) before Flow.frame(...) so HOLD "
                    "cannot duplicate or discard frames; contract=rate_before_frame"
                )
            if node.rate_policy is RatePolicy.SAMPLE:
                if node.rate_period is None:
                    raise RuntimeError("SAMPLE Node lacks period")
                _, input_period = signatures[source_port]
                ratio = node.rate_period / input_period
                if ratio.denominator != 1:
                    raise CompileError(
                        f"sample node {node.id} port {node.output_port} period "
                        f"{node.rate_period} is not an integer multiple of input period "
                        f"{input_period}; contract=stable_sample_boundary"
                    )
                proof = _TimingProof(
                    exact=source.exact,
                    contains_frame=source.contains_frame,
                    contains_rate=True,
                )
            else:
                proof = _TimingProof(exact=True, contains_frame=False, contains_rate=True)
        else:
            raise RuntimeError(f"unsupported Node kind {node.kind!r}")
        for output_port in node.output_ports:
            proofs[output_port] = proof


def _port_finiteness(nodes: Sequence[NodeSpec]) -> dict[int, bool]:
    """各Portの生成列が有限と証明できるかをGraph順に求める。"""

    finite: dict[int, bool] = {}
    for node in nodes:
        if node.kind is NodeKind.SOURCE:
            value = (
                False
                if isinstance(node.source, RealtimeSource)
                else node.source.is_finite
                if isinstance(node.source, Source)
                else True
            )
        else:
            value = all(finite[input_spec.source_port] for input_spec in node.inputs)
        for output_port in node.output_ports:
            finite[output_port] = value
    return finite


def _compile_diagnostics(nodes: Sequence[NodeSpec]) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    signatures = _time_signature(nodes)
    for node in nodes:
        if node.kind is NodeKind.MAP and node.time_transform == "explicit":
            diagnostics.append(
                Diagnostic(
                    Severity.INFO,
                    "EXPLICIT_TIME_TRANSFORM",
                    "Kernel defines output intervals; numerical resampling remains external",
                    node_id=node.id,
                    port_id=node.output_port,
                )
            )
        for path in node.config_paths or ():
            if not node.config.has(path):
                raise MissingConfigError(f"node {node.id} requires missing Config path {path!r}")
        if isinstance(node.operation, OperationDefinition):
            operation = node.operation
            scope = operation.spec.config.scope
            try:
                config_view = node.config.view(scope)
            except (KeyError, TypeError) as error:
                raise MissingConfigError(
                    f"node={node.id} port={node.output_port} "
                    f"operation={operation.operation_id} config_scope={scope!r} is unavailable"
                ) from error
            for path, expected in operation.spec.config.fields:
                try:
                    value = config_view.require(path)
                except KeyError as error:
                    raise MissingConfigError(
                        f"node={node.id} port={node.output_port} "
                        f"operation={operation.operation_id} config_scope={scope!r} "
                        f"missing_field={path!r}"
                    ) from error
                if not isinstance(value, expected):
                    expected_types = expected if isinstance(expected, tuple) else (expected,)
                    expected_name = "|".join(item.__name__ for item in expected_types)
                    raise MissingConfigError(
                        f"node={node.id} port={node.output_port} "
                        f"operation={operation.operation_id} config_field={path!r} "
                        f"expected={expected_name} actual={type(value).__name__}"
                    )
        sync_inputs = [item for item in node.inputs if item.semantics is InputSemantics.SYNCHRONOUS]
        if len(sync_inputs) < 2:
            continue
        input_signatures = {signatures[item.source_port] for item in sync_inputs}
        if len(input_signatures) > 1:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "POSSIBLE_INTERVAL_MISMATCH",
                    "synchronous inputs have different interval length or step",
                    node_id=node.id,
                    port_id=node.output_port,
                    details={"signatures": [str(value) for value in sorted(input_signatures)]},
                )
            )
    return tuple(diagnostics)


def _source_request_periods(nodes: Sequence[NodeSpec]) -> dict[int, Fraction]:
    """各Source配下で最短のRATE周期をpull request幅として求める。"""

    consumers: dict[int, list[NodeSpec]] = defaultdict(list)
    for node in nodes:
        for input_spec in node.inputs:
            consumers[input_spec.source_port].append(node)
    periods: dict[int, Fraction] = {}
    for source in (node for node in nodes if node.kind is NodeKind.SOURCE):
        pending = [source.output_port]
        visited: set[int] = set()
        candidates: list[Fraction] = []
        while pending:
            port = pending.pop()
            if port in visited:
                continue
            visited.add(port)
            for consumer in consumers.get(port, ()):
                if consumer.kind is NodeKind.RATE and consumer.rate_period is not None:
                    candidates.append(consumer.rate_period)
                pending.append(consumer.output_port)
        periods[source.id] = min(candidates, default=Fraction(1))
    return periods


def _node_max_items(
    node: NodeSpec,
    signatures: dict[int, tuple[Fraction, Fraction]],
) -> int:
    """一回のNode処理で生成し得るEmission件数上限を求める。"""

    if node.kind is not NodeKind.RATE or node.rate_policy is RatePolicy.SAMPLE:
        return node.max_items
    if node.rate_period is None:
        raise RuntimeError("RATE Node lacks period")
    input_duration = signatures[node.inputs[0].source_port][0]
    ratio = input_duration / node.rate_period
    return max(1, -(-ratio.numerator // ratio.denominator))


@dataclass(frozen=True)
class _BufferPlan:
    """一つのPORT_SHARED bufferについてcompileした上限と根拠。"""

    max_items: int
    high_watermark: int
    low_watermark: int
    capacity_reasons: tuple[str, ...]


def _planned_port_buffers(
    nodes: Sequence[NodeSpec],
    signatures: dict[int, tuple[Fraction, Fraction]],
) -> dict[int, _BufferPlan]:
    """merge分岐の共有祖先需要からPort別の保持上限を証明する。"""

    nodes_by_port = {port: node for node in nodes for port in node.output_ports}
    capacities = {
        port: _node_max_items(node, signatures) for node in nodes for port in node.output_ports
    }
    reasons: dict[int, list[str]] = {
        port: [f"producer_burst:node={node.id}:max_items={capacities[port]}"]
        for node in nodes
        for port in node.output_ports
    }
    consumer_counts: Counter[int] = Counter(
        input_spec.source_port for node in nodes for input_spec in node.inputs
    )
    demand_cache: dict[int, dict[int, int]] = {}

    def ancestor_demands(port_id: int) -> dict[int, int]:
        cached = demand_cache.get(port_id)
        if cached is not None:
            return cached
        node = nodes_by_port[port_id]
        demands = {port_id: 1}
        multiplier = 1
        if node.kind is NodeKind.FRAME:
            if node.frame_size is None:
                raise RuntimeError("FRAME Node lacks size")
            multiplier = node.frame_size
        elif node.kind is NodeKind.RATE and node.rate_policy is RatePolicy.SAMPLE:
            if node.rate_period is None:
                raise RuntimeError("SAMPLE Node lacks period")
            input_period = signatures[node.inputs[0].source_port][1]
            ratio = node.rate_period / input_period
            if ratio.denominator != 1:
                raise CompileError(f"sample node {node.id} period is not an integer input multiple")
            multiplier = ratio.numerator
        for input_spec in node.inputs:
            for ancestor, count in ancestor_demands(input_spec.source_port).items():
                demands[ancestor] = max(demands.get(ancestor, 0), count * multiplier)
        demand_cache[port_id] = demands
        return demands

    for node in nodes:
        if node.kind is not NodeKind.MAP or len(node.inputs) < 2:
            continue
        branch_demands = [ancestor_demands(item.source_port) for item in node.inputs]
        occurrence: Counter[int] = Counter(
            ancestor for branch in branch_demands for ancestor in branch
        )
        for ancestor, branch_count in occurrence.items():
            if branch_count < 2 or consumer_counts[ancestor] < 2:
                continue
            structural_required = max(branch.get(ancestor, 0) for branch in branch_demands)
            producer_burst = _node_max_items(nodes_by_port[ancestor], signatures)
            required = -(-structural_required // producer_burst) * producer_burst
            capacities[ancestor] = max(capacities[ancestor], required)
            reasons[ancestor].append(
                f"shared_merge_demand:node={node.id}:max_items={required}:"
                f"structural_items={structural_required}:producer_burst={producer_burst}"
            )

    return {
        port_id: _BufferPlan(
            max_items=capacity,
            high_watermark=capacity,
            low_watermark=max(0, capacity - 1),
            capacity_reasons=tuple(dict.fromkeys(reasons[port_id])),
        )
        for port_id, capacity in capacities.items()
    }


def _collector_descriptor(index: int, item: OutputSpec[Any]) -> OutputDescriptor:
    """collector instanceをportableな終端descriptorへ変換する。"""

    collector = item.collector
    if isinstance(collector, Bounded):
        kind, max_items, overflow = (
            "bounded",
            collector.max_items,
            collector.overflow.value,
        )
    elif isinstance(collector, Latest):
        kind, max_items, overflow = "latest", 1, None
    elif isinstance(collector, NoCollect):
        kind, max_items, overflow = "none", 0, None
    elif isinstance(collector, Sink):
        kind, max_items, overflow = "sink", None, None
    else:
        kind, max_items, overflow = "bound_collector", None, None
    return OutputDescriptor(
        index,
        item.flow.port_id,
        kind,
        max_items,
        overflow,
        f"collector:{index}",
    )


def _trigger_descriptor(observation: ObservationSpec) -> TriggerDescriptor:
    """公開TriggerをPython objectを含まないdescriptorへ変換する。"""

    trigger = observation.trigger
    if isinstance(trigger, Always):
        return TriggerDescriptor(trigger.kind, None, None, None)
    if isinstance(trigger, Every):
        return TriggerDescriptor(trigger.kind, trigger.count, None, None)
    if isinstance(trigger, EveryLogicalTime):
        return TriggerDescriptor(
            trigger.kind,
            None,
            RationalDescriptor.from_fraction(trigger.period),
            RationalDescriptor.from_fraction(trigger.offset),
        )
    raise TypeError(
        f"extension_id {observation.extension_id!r} port {observation.flow.port_id} "
        "uses an unsupported trigger contract"
    )


def _input_tolerance_descriptor(input_spec: InputSpec) -> RationalDescriptor | None:
    """InputSpecの任意toleranceをportable descriptorへ変換する。"""

    tolerance = input_spec.tolerance
    return None if tolerance is None else RationalDescriptor.from_fraction(tolerance)


def _stage_descriptors(
    nodes: tuple[NodeSpec, ...],
    node_backend_names: Mapping[int, str],
    boundary_ports: set[int],
) -> tuple[StageDescriptor, ...]:
    """Python callbackと観測境界を越えない最小Stage列を作る。"""

    stages: list[StageDescriptor] = []
    pending: list[int] = []
    pending_domain: str | None = None

    def finish(*reasons: str) -> None:
        nonlocal pending, pending_domain
        if not pending or pending_domain is None:
            return
        stages.append(
            StageDescriptor(
                len(stages),
                tuple(pending),
                pending_domain,
                tuple(dict.fromkeys(reasons or (pending_domain,))),
            )
        )
        pending = []
        pending_domain = None

    for node in nodes:
        if node.kind is NodeKind.SOURCE:
            finish("execution_domain_change")
            pending = [node.id]
            pending_domain = "python_source"
            reasons = ["python_source"]
            if any(port in boundary_ports for port in node.output_ports):
                reasons.append("observation_boundary")
            finish(*reasons)
            continue
        if node.kind is NodeKind.MAP:
            finish("execution_domain_change")
            pending = [node.id]
            pending_domain = node_backend_names[node.id]
            reasons = ["python_callback" if pending_domain == "python" else "kernel_binding"]
            if any(port in boundary_ports for port in node.output_ports):
                reasons.append("observation_boundary")
            finish(*reasons)
            continue
        if pending_domain != "executor_opcode":
            finish("execution_domain_change")
            pending_domain = "executor_opcode"
        pending.append(node.id)
        if any(port in boundary_ports for port in node.output_ports):
            finish("observation_boundary")
    finish("plan_end")
    return tuple(stages)


def _shape_product(shape: tuple[int, ...]) -> int:
    """固定shapeの要素数を求める。"""

    result = 1
    for item in shape:
        result *= item
    return result


def _value_strides(dtype: str | None, shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    """既知dtypeと固定shapeからcontiguous byte strideを求める。"""

    if shape is None or dtype is None:
        return None
    item_size = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}.get(dtype)
    if item_size is None:
        return None
    return tuple(item_size * _shape_product(shape[index + 1 :]) for index in range(len(shape)))


def _operation_shape_error(
    node: NodeSpec,
    *,
    input_name: str,
    expected: object,
    actual: object,
    dimension: int | None = None,
) -> ShapeMismatchError:
    operation = node.operation
    operation_id = (
        operation.operation_id if isinstance(operation, OperationDefinition) else "unknown"
    )
    suffix = "" if dimension is None else f" dimension={dimension}"
    return ShapeMismatchError(
        f"node={node.id} port={node.output_port} operation={operation_id} "
        f"input={input_name!r} expected={expected!r} actual={actual!r}{suffix}"
    )


def _resolve_operation_schemas(
    node: NodeSpec,
    descriptors: Mapping[str, ValueSchemaDescriptor],
    port_schemas: Mapping[int, str],
) -> tuple[ValueSchemaDescriptor, ...]:
    """OperationSpecで入力shapeをunifyし、全output schemaを解決する。"""

    operation = node.operation
    if not isinstance(operation, OperationDefinition):
        raise TypeError("operation schema resolution requires OperationDefinition")
    config_view = node.config.view(operation.spec.config.scope)
    declared_inputs = operation.spec.input_mapping
    actual_inputs: dict[str, ValueSchemaDescriptor] = {}
    symbols: dict[str, int] = {}
    for input_spec in node.inputs:
        if input_spec.keyword is None:
            raise CompileError(
                f"node={node.id} port={node.output_port} operation={operation.operation_id} "
                "contains an unnamed input"
            )
        name = input_spec.keyword
        actual = descriptors[port_schemas[input_spec.source_port]]
        expected = declared_inputs[name].value
        actual_inputs[name] = actual
        if expected.dtype is not None and actual.dtype != expected.dtype:
            raise _operation_shape_error(
                node,
                input_name=name,
                expected=expected.dtype,
                actual=actual.dtype,
            )
        if expected.device != actual.device:
            raise _operation_shape_error(
                node,
                input_name=name,
                expected=expected.device,
                actual=actual.device,
            )
        if expected.shape is None:
            continue
        if actual.shape is None or len(expected.shape) != len(actual.shape):
            raise _operation_shape_error(
                node,
                input_name=name,
                expected=expected.shape,
                actual=actual.shape,
            )
        for dimension_index, (expected_dimension, actual_dimension) in enumerate(
            zip(expected.shape, actual.shape, strict=True)
        ):
            if expected_dimension is None:
                continue
            if isinstance(expected_dimension, int):
                resolved = expected_dimension
            elif expected_dimension.startswith("$config."):
                path = expected_dimension.removeprefix("$config.")
                try:
                    value = config_view.require(path)
                except KeyError as error:
                    raise MissingConfigError(
                        f"node={node.id} port={node.output_port} "
                        f"operation={operation.operation_id} input={name!r} "
                        f"missing_shape_config={expected_dimension!r}"
                    ) from error
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise _operation_shape_error(
                        node,
                        input_name=name,
                        expected=expected_dimension,
                        actual=value,
                        dimension=dimension_index,
                    )
                resolved = value
            else:
                previous = symbols.setdefault(expected_dimension, actual_dimension)
                resolved = previous
            if resolved != actual_dimension:
                raise _operation_shape_error(
                    node,
                    input_name=name,
                    expected=resolved,
                    actual=actual_dimension,
                    dimension=dimension_index,
                )

    resolved_by_shape_resolver: tuple[int, ...] | None = None
    if operation.spec.shape_resolver is not None:
        if len(operation.spec.outputs) != 1:
            raise CompileError(
                f"node={node.id} port={node.output_port} operation={operation.operation_id} "
                "shape_resolver currently requires one output"
            )
        try:
            resolved_by_shape_resolver = operation.spec.shape_resolver(
                actual_inputs,
                config_view,
            )
        except Exception as error:
            raise CompileError(
                f"node={node.id} port={node.output_port} "
                f"operation={operation.operation_id} contract=shape_resolver_failed "
                f"error={type(error).__name__}: {error}"
            ) from error
        if not resolved_by_shape_resolver or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in resolved_by_shape_resolver
        ):
            raise CompileError(
                f"node={node.id} port={node.output_port} operation={operation.operation_id} "
                f"shape_resolver returned invalid shape {resolved_by_shape_resolver!r}"
            )

    primary = actual_inputs[operation.spec.primary_input_name]
    resolved_outputs: list[ValueSchemaDescriptor] = []
    for output_index, (_, output) in enumerate(operation.spec.outputs):
        if output.value == "same":
            resolved_outputs.append(primary)
            continue
        value_spec = output.value
        if not isinstance(value_spec, ValueSpec):
            raise RuntimeError("validated OperationOutputSpec lost its ValueSpec")
        shape: tuple[int, ...] | None
        if resolved_by_shape_resolver is not None:
            shape = resolved_by_shape_resolver
        elif value_spec.shape is None:
            shape = None
        else:
            resolved_dimensions: list[int] = []
            has_dynamic_dimension = False
            for dimension in value_spec.shape:
                if dimension is None:
                    has_dynamic_dimension = True
                    break
                if isinstance(dimension, int):
                    resolved_dimensions.append(dimension)
                elif dimension.startswith("$config."):
                    try:
                        value = config_view.require(dimension.removeprefix("$config."))
                    except KeyError as error:
                        raise MissingConfigError(
                            f"node={node.id} port={node.output_ports[output_index]} "
                            f"operation={operation.operation_id} "
                            f"missing_shape_config={dimension!r}"
                        ) from error
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise CompileError(
                            f"node={node.id} port={node.output_ports[output_index]} "
                            f"operation={operation.operation_id} output_config_dimension="
                            f"{dimension!r} actual={value!r}"
                        )
                    resolved_dimensions.append(value)
                elif dimension in symbols:
                    resolved_dimensions.append(symbols[dimension])
                else:
                    raise CompileError(
                        f"node={node.id} port={node.output_ports[output_index]} "
                        f"operation={operation.operation_id} unresolved_symbol={dimension!r}"
                    )
            shape = None if has_dynamic_dimension else tuple(resolved_dimensions)
        dtype = value_spec.dtype
        representation = value_spec.resolved_representation
        schema_id = (
            f"operation:{operation.operation_id}:{output_index}:"
            f"{dtype or 'opaque'}:{'dynamic' if shape is None else 'x'.join(map(str, shape))}"
        )
        resolved_outputs.append(
            ValueSchemaDescriptor(
                schema_id,
                representation,
                dtype,
                shape,
                _value_strides(dtype, shape),
                value_spec.device,
                value_spec.read_only,
            )
        )
    return tuple(resolved_outputs)


def _value_schema_descriptors(
    nodes: tuple[NodeSpec, ...],
    compiled_kernels: dict[int, CompiledKernel[object]],
) -> tuple[tuple[ValueSchemaDescriptor, ...], dict[int, str]]:
    """明示native契約だけをPort value schemaへ伝播する。"""

    opaque = ValueSchemaDescriptor("python:opaque", "python_opaque", None, None, None, "cpu", True)
    descriptors: dict[str, ValueSchemaDescriptor] = {opaque.value_schema_id: opaque}
    port_schemas: dict[int, str] = {}
    for node in nodes:
        schemas: tuple[ValueSchemaDescriptor, ...] = (opaque,) * len(node.output_ports)
        if node.kind is NodeKind.SOURCE and isinstance(node.source, F64SourceValues):
            schema = ValueSchemaDescriptor(
                "native:f64:scalar", "contiguous_f64", "float64", (), (), "cpu", True
            )
            schemas = (schema,)
        elif node.kind is NodeKind.SOURCE and isinstance(node.source, F64VectorSourceValues):
            schema = ValueSchemaDescriptor(
                f"native:f64:{node.source.width}",
                "contiguous_f64",
                "float64",
                (node.source.width,),
                (8,),
                "cpu",
                True,
            )
            schemas = (schema,)
        elif node.kind in {NodeKind.RATE, NodeKind.MAP}:
            input_schema = descriptors[port_schemas[node.inputs[0].source_port]]
            if isinstance(node.operation, OperationDefinition):
                schemas = _resolve_operation_schemas(node, descriptors, port_schemas)
            elif node.kind is NodeKind.RATE or isinstance(node.operation, IdentityF64Kernel):
                schemas = (input_schema,) * len(node.output_ports)
            elif (
                input_schema.representation == "contiguous_f64"
                and input_schema.shape is not None
                and isinstance(
                    provider := compiled_kernels.get(node.id),
                    NativeValueSchemaProvider,
                )
            ):
                output_shape = provider.resolve_output_shape(input_schema.shape)
                if not output_shape or any(item <= 0 for item in output_shape):
                    raise CompileError(
                        f"native Kernel node {node.id} port {node.output_port} returned "
                        "an invalid fixed output shape"
                    )
                strides = tuple(
                    8 * _shape_product(output_shape[index + 1 :])
                    for index in range(len(output_shape))
                )
                schema = ValueSchemaDescriptor(
                    "native:f64:" + "x".join(str(item) for item in output_shape),
                    "contiguous_f64",
                    provider.output_dtype,
                    output_shape,
                    strides,
                    "cpu",
                    True,
                )
                schemas = (schema,) * len(node.output_ports)
        elif node.kind is NodeKind.FRAME:
            input_schema = descriptors[port_schemas[node.inputs[0].source_port]]
            if input_schema.representation == "contiguous_f64" and node.frame_size is not None:
                shape = (node.frame_size, *(input_schema.shape or ()))
                strides = tuple(
                    8 * _shape_product(shape[index + 1 :]) for index in range(len(shape))
                )
                schema = ValueSchemaDescriptor(
                    "native:f64:" + "x".join(str(item) for item in shape),
                    "contiguous_f64",
                    "float64",
                    shape,
                    strides,
                    "cpu",
                    True,
                )
                schemas = (schema,)
        if len(schemas) != len(node.output_ports):
            raise CompileError(
                f"node {node.id} resolved {len(schemas)} schemas for "
                f"{len(node.output_ports)} output ports"
            )
        for output_port, schema in zip(node.output_ports, schemas, strict=True):
            descriptors.setdefault(schema.value_schema_id, schema)
            port_schemas[output_port] = schema.value_schema_id
    return tuple(descriptors.values()), port_schemas


def _portable_plan_ir(
    *,
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec[Any], ...],
    extensions: tuple[ObservationSpec, ...],
    diagnostics: tuple[Diagnostic, ...],
    backend_name: str,
    node_backend_names: dict[int, str],
    kernel_abi_descriptors: dict[int, KernelAbiDescriptor],
    compiled_kernels: dict[int, CompiledKernel[object]],
    source_request_periods: dict[int, Fraction],
) -> PortablePlanIR:
    """compile済みPython実体からportable descriptorだけを抽出する。"""

    signatures = _time_signature(nodes)
    buffer_plans = _planned_port_buffers(nodes, signatures)
    edge_rows = tuple(
        (node, input_index) for node in nodes for input_index, _ in enumerate(node.inputs)
    )
    edges = tuple(
        EdgeDescriptor(
            edge_id,
            node.inputs[input_index].source_port,
            node.id,
            input_index,
            node.inputs[input_index].semantics.value,
            node.inputs[input_index].keyword,
            node.inputs[input_index].source_port,
            edge_id,
            True,
            None,
            _input_tolerance_descriptor(node.inputs[input_index]),
            node.inputs[input_index].missing_policy.value,
        )
        for edge_id, (node, input_index) in enumerate(edge_rows)
    )
    cursors_by_port: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        cursors_by_port[edge.source_port_id].append(edge.cursor_id)

    plan_nodes = tuple(
        NodeDescriptor(
            node.id,
            node.kind.value,
            tuple(item.source_port for item in node.inputs),
            node.output_ports,
            node.config.scope_id,
            node_backend_names[node.id],
            (
                f"source:{node.id}"
                if node.kind is NodeKind.SOURCE
                else (
                    f"implementation:{node.id}"
                    if isinstance(node.operation, OperationDefinition)
                    else f"kernel:{node.id}"
                )
                if node.kind is NodeKind.MAP
                else None
            ),
            node.accepts_invalid,
            node.output_port,
            _node_max_items(node, signatures),
            node.frame_size,
            node.frame_hop,
            node.pad_end,
            (
                None
                if node.rate_period is None
                else RationalDescriptor.from_fraction(node.rate_period)
            ),
            None if node.rate_policy is None else node.rate_policy.value,
            node.time_transform,
            node.gap_policy.value,
        )
        for node in nodes
    )
    value_schemas, port_schema_ids = _value_schema_descriptors(nodes, compiled_kernels)
    operation_descriptors: list[OperationDescriptor] = []
    implementation_descriptors: list[ImplementationDescriptor] = []
    for node in nodes:
        operation = node.operation
        if not isinstance(operation, OperationDefinition):
            continue
        compiled = compiled_kernels[node.id]
        if not isinstance(compiled, CompiledOperationMetadata):
            raise CompileError(
                f"node={node.id} port={node.output_port} operation={operation.operation_id} "
                "contract=compiled_operation_metadata"
            )
        implementation = compiled.implementation_spec
        if implementation.operation_id != operation.operation_id:
            raise CompileError(
                f"node={node.id} port={node.output_port} operation={operation.operation_id} "
                f"implementation_operation={implementation.operation_id} "
                "contract=operation_id_match"
            )
        slot = f"implementation:{node.id}"
        implementation_descriptors.append(
            ImplementationDescriptor(
                implementation.operation_id,
                implementation.implementation_id,
                implementation.backend,
                implementation.abi_version,
                slot,
                implementation.process_model,
                implementation.native_compatible,
                implementation.selected_variant,
                implementation.required_cpu_features,
                implementation.workspace_size_bytes,
                implementation.workspace_alignment_bytes,
                implementation.supports_flush,
                implementation.session_local,
            )
        )
        inputs_by_name = {item.keyword: item for item in node.inputs}
        operation_descriptors.append(
            OperationDescriptor(
                node.id,
                operation.operation_id,
                tuple(
                    OperationInputDescriptor(
                        name,
                        (inputs_by_name[name].source_port if name in inputs_by_name else None),
                        (
                            port_schema_ids[inputs_by_name[name].source_port]
                            if name in inputs_by_name
                            else None
                        ),
                        spec.mode,
                        spec.required,
                        spec.primary,
                    )
                    for name, spec in operation.spec.inputs
                ),
                tuple(
                    OperationOutputDescriptor(
                        name,
                        node.output_ports[index],
                        port_schema_ids[node.output_ports[index]],
                        spec.time,
                        spec.emissions,
                        spec.max_items,
                    )
                    for index, (name, spec) in enumerate(operation.spec.outputs)
                ),
                operation.spec.config.scope,
                node.config.scope_id,
                node.config.digest,
                tuple(
                    ConfigFieldDescriptor(
                        path,
                        tuple(
                            item.__module__ + "." + item.__qualname__
                            for item in (expected if isinstance(expected, tuple) else (expected,))
                        ),
                    )
                    for path, expected in operation.spec.config.fields
                ),
                operation.spec.state,
                operation.spec.gap_policy.value,
                operation.spec.accepts_invalid,
                "preserve_worst",
                implementation.implementation_id,
                implementation.abi_version,
                node_backend_names[node.id],
                slot,
            )
        )
    ports = tuple(
        PortDescriptor(
            output_port,
            node.id,
            output_index,
            port_schema_ids[output_port],
            output_port,
            f"port:{output_port}",
            output_port,
        )
        for node in nodes
        for output_index, output_port in enumerate(node.output_ports)
    )
    port_buffers = tuple(
        BufferDescriptor(
            buffer_id=output_port,
            kind="port_shared",
            producer_port_id=output_port,
            owner_node_id=None,
            owner_input_index=None,
            consumer_cursor_ids=tuple(cursors_by_port[output_port]),
            max_items=buffer_plans[output_port].max_items,
            max_bytes=None,
            capacity_reasons=buffer_plans[output_port].capacity_reasons,
            high_watermark=buffer_plans[output_port].high_watermark,
            low_watermark=buffer_plans[output_port].low_watermark,
            overflow_policy="fail",
            reclaim_policy="all_consumers_advanced",
            read_only=True,
            device="cpu",
            alignment_bytes=None,
            ownership="executor",
            copy_policy="shared_reference",
        )
        for node in nodes
        for output_port in node.output_ports
    )
    edge_by_input = {(edge.target_node_id, edge.target_input_index): edge for edge in edges}
    next_buffer_id = max((port for node in nodes for port in node.output_ports), default=-1) + 1
    internal_buffers: list[BufferDescriptor] = []
    adapter_by_edge: dict[int, int] = {}
    for node in nodes:
        if node.kind is NodeKind.SOURCE and isinstance(node.source, RealtimeSource):
            if (
                isinstance(node.source.max_items, bool)
                or not isinstance(node.source.max_items, int)
                or node.source.max_items <= 0
            ):
                raise CompileError(
                    f"realtime source node {node.id} port {node.output_port} requires "
                    "positive max_items"
                )
            if not isinstance(node.source.overflow_policy, RealtimeOverflowPolicy):
                raise CompileError(
                    f"realtime source node {node.id} port {node.output_port} has invalid "
                    "overflow_policy; contract=RealtimeOverflowPolicy"
                )
            internal_buffers.append(
                BufferDescriptor(
                    buffer_id=next_buffer_id,
                    kind="realtime_ingress",
                    producer_port_id=node.output_port,
                    owner_node_id=node.id,
                    owner_input_index=None,
                    consumer_cursor_ids=(),
                    max_items=node.source.max_items,
                    max_bytes=None,
                    capacity_reasons=(
                        f"realtime_ingress:node={node.id}:max_items={node.source.max_items}",
                    ),
                    high_watermark=node.source.max_items,
                    low_watermark=max(0, node.source.max_items - 1),
                    overflow_policy=node.source.overflow_policy.value,
                    reclaim_policy="scheduler_take",
                    read_only=True,
                    device="cpu",
                    alignment_bytes=None,
                    ownership="executor",
                    copy_policy="shared_reference",
                )
            )
            next_buffer_id += 1
        if node.kind is NodeKind.FRAME:
            if node.frame_size is None or node.frame_hop is None:
                raise RuntimeError("FRAME Node lacks size or hop")
            edge = edge_by_input[(node.id, 0)]
            internal_buffers.append(
                BufferDescriptor(
                    buffer_id=next_buffer_id,
                    kind="frame_history",
                    producer_port_id=node.inputs[0].source_port,
                    owner_node_id=node.id,
                    owner_input_index=0,
                    consumer_cursor_ids=(edge.cursor_id,),
                    max_items=node.frame_size,
                    max_bytes=None,
                    capacity_reasons=(
                        f"frame_history:node={node.id}:size={node.frame_size}:hop={node.frame_hop}",
                    ),
                    high_watermark=node.frame_size,
                    low_watermark=max(0, node.frame_size - 1),
                    overflow_policy="fail",
                    reclaim_policy="frame_hop",
                    read_only=True,
                    device="cpu",
                    alignment_bytes=None,
                    ownership="executor",
                    copy_policy="shared_reference",
                )
            )
            adapter_by_edge[edge.edge_id] = next_buffer_id
            next_buffer_id += 1
        for input_index, input_spec in enumerate(node.inputs):
            edge = edge_by_input[(node.id, input_index)]
            if input_spec.semantics is InputSemantics.LATEST:
                kind = "latest_state"
                reason = f"latest_state:node={node.id}:input={input_index}:max_items=1"
                overflow = "replace_oldest"
                reclaim = "replace_on_newer"
            elif input_spec.semantics in {
                InputSemantics.CONTAINS,
                InputSemantics.OVERLAPS,
                InputSemantics.TOLERANCE,
            }:
                kind = "sync_selection"
                reason = f"sync_selection:node={node.id}:input={input_index}:max_items=1"
                overflow = "replace_when_before_frontier"
                reclaim = "reference_interval_advanced"
            else:
                continue
            internal_buffers.append(
                BufferDescriptor(
                    buffer_id=next_buffer_id,
                    kind=kind,
                    producer_port_id=input_spec.source_port,
                    owner_node_id=node.id,
                    owner_input_index=input_index,
                    consumer_cursor_ids=(edge.cursor_id,),
                    max_items=1,
                    max_bytes=None,
                    capacity_reasons=(reason,),
                    high_watermark=1,
                    low_watermark=0,
                    overflow_policy=overflow,
                    reclaim_policy=reclaim,
                    read_only=True,
                    device="cpu",
                    alignment_bytes=None,
                    ownership="executor",
                    copy_policy="shared_reference",
                )
            )
            adapter_by_edge[edge.edge_id] = next_buffer_id
            next_buffer_id += 1
    edges = tuple(
        replace(edge, adapter_buffer_id=adapter_by_edge.get(edge.edge_id)) for edge in edges
    )
    buffers = port_buffers + tuple(internal_buffers)
    port_finiteness = _port_finiteness(nodes)
    times = tuple(
        TimeDescriptor(
            output_port,
            RationalDescriptor(1, 1),
            RationalDescriptor.from_fraction(signatures[output_port][0]),
            RationalDescriptor.from_fraction(signatures[output_port][1]),
            RationalDescriptor(0, 1),
            node.time_transform if node.kind is NodeKind.MAP else node.kind.value,
            not (node.kind is NodeKind.MAP and node.time_transform == "explicit"),
            port_finiteness[output_port],
            None,
        )
        for node in nodes
        for output_port in node.output_ports
    )
    ingress_by_node = {
        item.owner_node_id: item for item in buffers if item.kind == "realtime_ingress"
    }
    sources = tuple(
        SourceDescriptor(
            node_id=node.id,
            mode=(
                "realtime_push" if isinstance(node.source, RealtimeSource) else "pull_controlled"
            ),
            is_finite=(
                False
                if isinstance(node.source, RealtimeSource)
                else node.source.is_finite
                if isinstance(node.source, Source)
                else True
            ),
            request_duration=RationalDescriptor.from_fraction(source_request_periods[node.id]),
            burst_max_items=(
                node.source.max_items if isinstance(node.source, RealtimeSource) else None
            ),
            ingress_buffer_id=(
                ingress_by_node[node.id].buffer_id
                if isinstance(node.source, RealtimeSource)
                else None
            ),
            overflow_policy=(
                node.source.overflow_policy.value
                if isinstance(node.source, RealtimeSource)
                else None
            ),
            gap_policy="degrade_next",
        )
        for node in nodes
        if node.kind is NodeKind.SOURCE
    )
    extension_descriptors = tuple(
        ExtensionDescriptor(
            extension.extension_id,
            extension.flow.port_id,
            _trigger_descriptor(extension),
            extension.priority,
            extension.failure_policy.value,
            extension.overflow_policy.value,
            f"extension:{extension.extension_id}",
            extension.abi_version,
        )
        for extension in extensions
    )
    bindings = tuple(
        [
            BindingDescriptor(f"source:{node.id}", "source", node.id, node.output_port, "python-v1")
            for node in nodes
            if node.kind is NodeKind.SOURCE
        ]
        + [
            BindingDescriptor(
                (
                    f"implementation:{node.id}"
                    if isinstance(node.operation, OperationDefinition)
                    else f"kernel:{node.id}"
                ),
                "operation" if isinstance(node.operation, OperationDefinition) else "kernel",
                node.id,
                node.output_port,
                kernel_abi_descriptors[node.id].abi_version,
            )
            for node in nodes
            if node.kind is NodeKind.MAP
        ]
        + [
            BindingDescriptor(
                f"collector:{index}",
                "collector",
                None,
                item.flow.port_id,
                "collector-v1",
            )
            for index, item in enumerate(outputs)
        ]
        + [
            BindingDescriptor(
                extension.binding_slot,
                "extension",
                None,
                extension.observed_port_id,
                extension.abi_version,
            )
            for extension in extension_descriptors
        ]
    )
    boundary_ports = {item.flow.port_id for item in outputs} | {
        item.observed_port_id for item in extension_descriptors
    }
    stages = _stage_descriptors(nodes, node_backend_names, boundary_ports)
    kernel_abis = tuple(
        kernel_abi_descriptors[node.id] for node in nodes if node.kind is NodeKind.MAP
    )
    native_port_ids = tuple(
        port.port_id for port in ports if port.value_schema_id != "python:opaque"
    )
    stream_item_abis = (
        (
            StreamItemAbiDescriptor(
                "chronowire.stream_item.v1",
                "structure_of_arrays",
                "signed_i64_ticks_shared_rational_timebase",
                "implicit_port_monotonic_u64",
                "u8:ok=0,degraded=1,invalid=2",
                "source_provenance_index_table",
                "python_boundary_object_table",
            ),
        )
        if native_port_ids
        else ()
    )
    native_buffers = tuple(
        NativeBufferDescriptor(
            port.port_id,
            port.value_schema_id,
            "chronowire.stream_item.v1",
            "structure_of_arrays",
            8,
            "session_owned_immutable_shared",
            True,
        )
        for port in ports
        if port.port_id in native_port_ids
    )
    return PortablePlanIR(
        schema_version="0.4" if operation_descriptors else "0.3",
        kind="execution_plan",
        backend=backend_name,
        nodes=plan_nodes,
        value_schemas=value_schemas,
        stages=stages,
        kernel_abis=kernel_abis,
        operations=tuple(operation_descriptors),
        implementations=tuple(implementation_descriptors),
        stream_item_abis=stream_item_abis,
        native_buffers=native_buffers,
        ports=ports,
        edges=edges,
        buffers=buffers,
        times=times,
        sources=sources,
        extensions=extension_descriptors,
        bindings=bindings,
        outputs=tuple(_collector_descriptor(index, item) for index, item in enumerate(outputs)),
        diagnostics=tuple(
            PlanDiagnosticDescriptor(
                item.severity.value, item.code, item.message, item.node_id, item.port_id
            )
            for item in diagnostics
        ),
    )


def compile(
    outputs: Sequence[Flow[Any] | OutputSpec[Any]],
    *,
    backend: str | Backend = "python",
    extensions: Sequence[ObservationSpec] = (),
) -> ExecutionPlan:
    """Flow群から不変なExecutionPlanを生成する。

    Args:
        outputs: 観測終端。bare FlowはNoCollectとして実行だけ行う。
        extensions: `observe()`で固定したcompile-time観測契約。

    Raises:
        ValueError: outputsが空、または異なるGraphを含む場合。
        CompileError: rate/frame格子を静的に証明できない場合。
        DuplicateOutputError: 同じPortが複数回指定された場合。
        DuplicateExtensionIdError: extension_idが重複した場合。
        MissingConfigError: 宣言されたConfig pathが存在しない場合。
    """

    if not outputs:
        raise ValueError("compile outputs must not be empty")
    normalized: list[OutputSpec[Any]] = []
    for item in outputs:
        normalized.append(output(item, collector=NoCollect()) if isinstance(item, Flow) else item)

    graph = normalized[0].flow._graph
    if any(item.flow._graph is not graph for item in normalized):
        raise ValueError("all compile outputs must belong to the same Graph")
    ports = [item.flow.port_id for item in normalized]
    if len(set(ports)) != len(ports):
        raise DuplicateOutputError("compile output ports must be unique")

    extension_ids: dict[str, ObservationSpec] = {}
    for extension in extensions:
        if not isinstance(extension, ObservationSpec):
            raise CompileError(
                "compile extensions must be ObservationSpec values returned by observe(); "
                "bind Extension handlers with ExecutionPlan.create_session()"
            )
        if extension.flow._graph is not graph:
            raise ValueError(
                f"extension_id {extension.extension_id!r} port {extension.flow.port_id} "
                "belongs to a different Graph"
            )
        previous = extension_ids.get(extension.extension_id)
        if previous is not None:
            previous_node = graph.node_for_port(previous.flow.port_id)
            current_node = graph.node_for_port(extension.flow.port_id)
            raise DuplicateExtensionIdError(
                f"duplicate extension_id {extension.extension_id!r} "
                f"slot 'extension:{extension.extension_id}'; nodes {previous_node.id} and "
                f"{current_node.id}; ports {previous.flow.port_id} and "
                f"{extension.flow.port_id}; contract=unique_extension_id"
            )
        extension_ids[extension.extension_id] = extension

    observed_ports = [extension.flow.port_id for extension in extensions]
    roots = tuple(ports + observed_ports)
    nodes = _required_nodes(graph, roots)
    _validate_rate_frame_boundaries(nodes)
    diagnostics = _compile_diagnostics(nodes)
    # Operationの言語非依存意味論は実装選択より先に確定する。legacy Kernelの
    # native schema providerだけは後段の互換passで補完する。
    _value_schema_descriptors(nodes, {})
    backend_instance: Backend
    if isinstance(backend, str):
        if backend != "python":
            raise ValueError(f"unsupported backend {backend!r}")
        backend_instance = PythonBackend()
    else:
        backend_instance = backend
    compiled_kernels: dict[int, CompiledKernel[object]] = {}
    kernel_abi_descriptors: dict[int, KernelAbiDescriptor] = {}
    node_backend_names: dict[int, str] = {
        node.id: "executor" for node in nodes if node.kind is not NodeKind.MAP
    }
    python_backend = PythonBackend()
    for node in nodes:
        if node.kind is not NodeKind.MAP:
            continue
        operation = node.operation
        selected_backend = backend_instance
        kernel: Kernel[object]
        if isinstance(operation, OperationDefinition):
            input_names = tuple(item.keyword for item in node.inputs if item.keyword is not None)
            if len(input_names) != len(node.inputs):
                raise CompileError(
                    f"node={node.id} port={node.output_port} "
                    f"operation={operation.operation_id} contains unnamed inputs"
                )
            if selected_backend.name == "python":
                if (
                    operation.python_binding is None
                    or operation.python_binding.spec.backend != "python"
                ):
                    raise MissingImplementationError(
                        f"node={node.id} port={node.output_port} "
                        f"operation={operation.operation_id} backend=python "
                        "contract=missing_implementation"
                    )
                compiled = compile_python_operation(
                    operation,
                    input_names=input_names,
                    config=node.config.view(operation.spec.config.scope),
                )
            elif isinstance(selected_backend, OperationBackend):
                compiled = selected_backend.compile_operation(
                    operation.spec,
                    CompileContext(node.config, node.constants or {}),
                )
            else:
                raise MissingImplementationError(
                    f"node={node.id} port={node.output_port} "
                    f"operation={operation.operation_id} backend={selected_backend.name} "
                    "contract=missing_implementation"
                )
            if not isinstance(compiled, CompiledKernel):
                raise TypeError(
                    f"backend {selected_backend.name!r} returned an invalid "
                    f"CompiledOperation for node {node.id} operation={operation.operation_id}"
                )
            if not isinstance(compiled, CompiledOperationMetadata):
                raise CompileError(
                    f"node={node.id} port={node.output_port} "
                    f"operation={operation.operation_id} "
                    "contract=compiled_operation_metadata"
                )
            implementation = compiled.implementation_spec
            if (
                implementation.operation_id != operation.operation_id
                or implementation.backend != selected_backend.name
            ):
                raise CompileError(
                    f"node={node.id} port={node.output_port} "
                    f"operation={operation.operation_id} "
                    f"implementation_operation={implementation.operation_id} "
                    f"implementation_backend={implementation.backend} "
                    f"backend={selected_backend.name} contract=implementation_identity"
                )
            compiled_kernels[node.id] = compiled
            if isinstance(compiled, NativeCompiledKernel):
                if implementation.abi_version != compiled.abi_version:
                    raise CompileError(
                        f"node={node.id} port={node.output_port} "
                        f"operation={operation.operation_id} "
                        f"implementation_abi={implementation.abi_version} "
                        f"compiled_abi={compiled.abi_version} contract=abi_match"
                    )
                kernel_abi_descriptors[node.id] = KernelAbiDescriptor(
                    node.id,
                    f"implementation:{node.id}",
                    compiled.abi_version,
                    compiled.process_model,
                    compiled.workspace_size_bytes,
                    compiled.workspace_alignment_bytes,
                    compiled.supports_flush,
                    compiled.session_local,
                    compiled.native_compatible,
                )
            else:
                kernel_abi_descriptors[node.id] = KernelAbiDescriptor(
                    node.id,
                    f"implementation:{node.id}",
                    implementation.abi_version,
                    implementation.process_model,
                    implementation.workspace_size_bytes,
                    implementation.workspace_alignment_bytes,
                    implementation.supports_flush,
                    implementation.session_local,
                    implementation.native_compatible,
                )
            node_backend_names[node.id] = selected_backend.name
            continue
        if isinstance(operation, Kernel):
            kernel = operation
        elif callable(operation):
            try:
                inject_config = "config" in inspect.signature(operation).parameters
            except (TypeError, ValueError):
                inject_config = False
            keywords = tuple(item.keyword for item in node.inputs[1:] if item.keyword is not None)
            if len(keywords) != len(node.inputs) - 1:
                raise ValueError(f"node {node.id} has an unnamed Python callable input")
            kernel = PythonCallableKernel(operation, keywords, inject_config)
            selected_backend = python_backend
        else:
            raise TypeError(f"MAP node {node.id} lacks a Kernel or Python callable")
        compiled = selected_backend.compile_kernel(
            kernel,
            CompileContext(node.config, node.constants or {}),
        )
        if not isinstance(compiled, CompiledKernel):
            raise TypeError(
                f"backend {selected_backend.name!r} returned an invalid "
                f"CompiledKernel for node {node.id}"
            )
        compiled_kernels[node.id] = compiled
        if isinstance(compiled, NativeCompiledKernel):
            kernel_abi_descriptors[node.id] = KernelAbiDescriptor(
                node.id,
                f"kernel:{node.id}",
                compiled.abi_version,
                compiled.process_model,
                compiled.workspace_size_bytes,
                compiled.workspace_alignment_bytes,
                compiled.supports_flush,
                compiled.session_local,
                compiled.native_compatible,
            )
        elif isinstance(operation, IdentityF64Kernel):
            kernel_abi_descriptors[node.id] = KernelAbiDescriptor(
                node.id,
                f"kernel:{node.id}",
                operation.abi_version,
                operation.process_model,
                0,
                8,
                False,
                True,
                True,
            )
        else:
            kernel_abi_descriptors[node.id] = KernelAbiDescriptor(
                node.id,
                f"kernel:{node.id}",
                "python-v1",
                "python_object",
                None,
                None,
                False,
                True,
                False,
            )
        node_backend_names[node.id] = (
            "native_kernel" if isinstance(operation, IdentityF64Kernel) else selected_backend.name
        )
    source_request_periods = _source_request_periods(nodes)
    sorted_extensions = tuple(sorted(extensions, key=lambda item: item.priority))
    portable_ir = _portable_plan_ir(
        nodes=nodes,
        outputs=tuple(normalized),
        extensions=sorted_extensions,
        diagnostics=diagnostics,
        backend_name=backend_instance.name,
        node_backend_names=node_backend_names,
        kernel_abi_descriptors=kernel_abi_descriptors,
        compiled_kernels=compiled_kernels,
        source_request_periods=source_request_periods,
    )
    return ExecutionPlan(
        graph=graph,
        nodes=nodes,
        outputs=tuple(normalized),
        observations=sorted_extensions,
        compile_diagnostics=diagnostics,
        compiled_kernels=compiled_kernels,
        backend_name=backend_instance.name,
        node_backend_names=node_backend_names,
        source_request_periods=source_request_periods,
        portable_ir=portable_ir,
    )


class ExecutionPlan:
    """compile後のrequired Nodeと実行policyを保持する不変な計画。

    Graph構造は変更せず、各runでcollector、buffer、KernelStateを作り直す。
    """

    def __init__(
        self,
        *,
        graph: Graph,
        nodes: tuple[NodeSpec, ...],
        outputs: tuple[OutputSpec[Any], ...],
        observations: tuple[ObservationSpec, ...],
        compile_diagnostics: tuple[Diagnostic, ...],
        compiled_kernels: dict[int, CompiledKernel[object]],
        backend_name: str,
        node_backend_names: dict[int, str],
        source_request_periods: dict[int, Fraction],
        portable_ir: PortablePlanIR,
    ) -> None:
        self._graph = graph
        self._nodes = nodes
        self._outputs = outputs
        self._observations = observations
        self._compile_diagnostics = compile_diagnostics
        self._compiled_kernels = compiled_kernels
        self._backend_name = backend_name
        self._node_backend_names = node_backend_names
        self._source_request_periods = source_request_periods
        self._portable_ir = portable_ir

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """compile時に生成したwarningを返す。"""

        return self._compile_diagnostics

    @property
    def portable_ir(self) -> PortablePlanIR:
        """Executor非依存の不変なExecutionPlan descriptorを返す。"""

        return self._portable_ir

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
        executor: str | Executor = "python",
    ) -> RunResult:
        """単一threadの決定的SchedulerでPlanを実行する。

        Args:
            duration: Sourceの論理時間上限。Noneではfinite SourceのEOFまで実行。
            options: Source chunk、watermark、budget、profiler設定。
            executor: 実行sessionを生成するExecutor名または実体。

        Returns:
            collector結果、Diagnostic、status件数を持つRunResult。
        """

        return self.create_session(executor=executor).run(duration=duration, options=options)

    def create_session(
        self,
        *,
        extension_bindings: Mapping[str, Extension] | None = None,
        executor: str | Executor = "python",
    ) -> ExecutorSession:
        """process-local Extension実体を検証して実行instanceを生成する。

        Args:
            extension_bindings: extension_idからExtension factoryへの完全な対応。
            executor: 実行sessionを生成するExecutor名または実体。

        Returns:
            同じPlanをrun-local状態で実行するExecutionSession。

        Raises:
            ExtensionBindingError: binding不足、未知ID、種別、ABI不整合の場合。
        """

        return self._resolve_executor(executor).create_session(self, extension_bindings)

    def _create_python_session(
        self,
        extension_bindings: Mapping[str, Extension] | None,
    ) -> ExecutionSession:
        """検証済みExtensionを持つPython実行sessionを生成する。"""

        bindings = {} if extension_bindings is None else dict(extension_bindings)
        required = {item.extension_id: item for item in self._observations}
        missing = [item for key, item in required.items() if key not in bindings]
        if missing:
            item = missing[0]
            node = self._graph.node_for_port(item.flow.port_id)
            raise ExtensionBindingError(
                f"extension_id {item.extension_id!r} slot 'extension:{item.extension_id}' "
                f"node {node.id} port {item.flow.port_id} missing required binding; "
                "contract=required_extension_binding"
            )
        unknown = sorted(set(bindings) - set(required))
        if unknown:
            extension_id = unknown[0]
            raise ExtensionBindingError(
                f"extension_id {extension_id!r} slot 'extension:{extension_id}' "
                "node None port None is unknown or unused; contract=known_extension_binding"
            )
        bound: list[_BoundExtension] = []
        for item in self._observations:
            binding = bindings[item.extension_id]
            node = self._graph.node_for_port(item.flow.port_id)
            slot = f"extension:{item.extension_id}"
            if not isinstance(binding, Extension):
                raise ExtensionBindingError(
                    f"extension_id {item.extension_id!r} slot {slot!r} node {node.id} "
                    f"port {item.flow.port_id} has invalid binding type; "
                    "contract=Extension.create_session"
                )
            if binding.abi_version != item.abi_version:
                raise ExtensionBindingError(
                    f"extension_id {item.extension_id!r} slot {slot!r} node {node.id} "
                    f"port {item.flow.port_id} ABI {binding.abi_version!r} does not match "
                    f"required {item.abi_version!r}; contract=extension_abi"
                )
            bound.append(_BoundExtension(item, binding))
        return ExecutionSession(self, tuple(bound))

    def create_plan_session(
        self,
        *,
        extension_bindings: Mapping[str, Extension] | None = None,
        options: RuntimeOptions | None = None,
        executor: str | Executor = "python",
    ) -> ExecutorPlanSession:
        """v0.2の継続実行状態を持つPlanSessionを生成する。

        Args:
            extension_bindings: extension_idからrun-local Extension factoryへの完全な対応。
            options: session全体へ適用するruntime調整値。
            executor: 継続sessionを生成するExecutor名または実体。

        Returns:
            `start()`前の新しいPlanSession。

        Raises:
            ExtensionBindingError: binding不足、未知ID、種別、ABI不整合の場合。
        """

        return self._resolve_executor(executor).create_plan_session(
            self,
            extension_bindings,
            options,
        )

    def _create_python_plan_session(
        self,
        extension_bindings: Mapping[str, Extension] | None,
        options: RuntimeOptions | None,
    ) -> PlanSession:
        """検証済みExtensionを持つPython継続sessionを生成する。"""

        one_shot = self._create_python_session(extension_bindings)
        return PlanSession(self, one_shot._extensions, options or RuntimeOptions())

    @staticmethod
    def _resolve_executor(executor: str | Executor) -> Executor:
        """公開指定を検証済みExecutorへ正規化する。"""

        if executor == "python":
            return PythonExecutor()
        if executor == "cython":
            return CythonExecutor()
        if executor == "cpp":
            return CppExecutor()
        if isinstance(executor, str):
            raise ValueError(f"unsupported executor {executor!r}")
        if not isinstance(executor, Executor):
            raise TypeError("executor must implement the Executor protocol")
        return executor

    def export(self, path: str | Path) -> None:
        """required Node、output、compile DiagnosticをJSONまたはDOTへ出力する。

        Raises:
            ValueError: 拡張子が`.json`または`.dot`でない場合。
        """

        output_path = Path(path)
        if output_path.suffix == ".json":
            output_path.write_text(self._portable_ir.to_json(), encoding="utf-8")
            return
        if output_path.suffix == ".dot":
            required = {node.id for node in self._nodes}
            lines = ["digraph chronowire_plan {"]
            lines.extend(
                f'  n{node.id} [label="{node.id}: {node.kind.value}"];' for node in self._nodes
            )
            for node in self._nodes:
                for input_spec in node.inputs:
                    source = self._graph.node_for_port(input_spec.source_port)
                    if source.id in required:
                        lines.append(f"  n{source.id} -> n{node.id};")
            lines.append("}")
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        raise ValueError("ExecutionPlan export supports only .json and .dot")


@dataclass(frozen=True)
class _BoundExtension:
    observation: ObservationSpec
    binding: Extension


@dataclass
class _ExtensionRuntime:
    observation: ObservationSpec
    session: ExtensionSession
    trigger: TriggerSession


class ExecutionSession:
    """ExecutionPlanと検証済みprocess-local bindingを結ぶ実行instance。"""

    def __init__(
        self,
        plan: ExecutionPlan,
        extensions: tuple[_BoundExtension, ...],
    ) -> None:
        self._plan = plan
        self._extensions = extensions

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """新しいKernel、collector、Extension状態でPlanを一回実行する。

        Args:
            duration: Sourceの論理時間上限。Noneではfinite SourceのEOFまで実行。
            options: Source chunk、watermark、budget、profiler設定。

        Returns:
            collector結果、Diagnostic、status件数を持つRunResult。
        """

        runtime = _PlanRuntime(
            self._plan,
            duration,
            self._extensions,
            options=options or RuntimeOptions(),
        )
        return runtime.run()


class PlanSessionState(StrEnum):
    """PlanSessionの公開lifecycle状態。"""

    CREATED = "created"
    RUNNING = "running"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PlanSession:
    """一つのExecutionPlanを論理時間境界ごとに継続実行するsession。

    Kernel、FRAME、RATE、buffer、collector、Extensionの状態はsession終了まで保持し、
    別のPlanSessionとは共有しない。
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        extensions: tuple[_BoundExtension, ...],
        options: RuntimeOptions,
    ) -> None:
        self._plan = plan
        self._extensions = extensions
        self._options = options
        self._runtime: _PlanRuntime | None = None
        self._state = PlanSessionState.CREATED
        self._logical_end: Fraction | None = None

    @property
    def state(self) -> PlanSessionState:
        """現在のlifecycle状態を返す。"""

        return self._state

    def start(self) -> None:
        """run-local resourceを生成して継続実行を開始する。

        Raises:
            PlanSessionError: CREATED以外から開始しようとした場合。
        """

        if self._state is not PlanSessionState.CREATED:
            raise PlanSessionError(
                f"PlanSession.start requires state=created; actual={self._state.value}"
            )
        runtime = _PlanRuntime(
            self._plan,
            None,
            self._extensions,
            continuous=True,
            options=self._options,
        )
        try:
            runtime.start()
        except Exception as error:
            self._fail_and_finish(runtime, error)
            raise
        self._runtime = runtime
        self._state = PlanSessionState.RUNNING

    def run_until(
        self,
        logical_end: LogicalTime | Fraction | int | float,
    ) -> RunResult:
        """session状態を保持したまま指定論理時刻まで進める。

        Args:
            logical_end: Emission終端の論理時間上限。前回指定値より大きい正数。

        Returns:
            session開始時から現在境界までの累積RunResult snapshot。

        Raises:
            PlanSessionError: 未開始、終了済み、または境界が単調増加でない場合。
        """

        runtime = self._running_runtime("run_until")
        target = self._logical_time_fraction(logical_end)
        if target <= 0 or (self._logical_end is not None and target <= self._logical_end):
            raise PlanSessionError(
                "PlanSession.run_until requires a positive, strictly increasing logical_end"
            )
        try:
            result = runtime.run_until(target)
        except Exception as error:
            self._fail_and_finish(runtime, error)
            raise
        if not runtime.last_budget_exhausted:
            self._logical_end = target
        return result

    def flush(self) -> RunResult:
        """有限SourceをEOFまで進め、FRAME等のpending状態をdrainする。

        Returns:
            flush後の累積RunResult snapshot。

        Raises:
            PlanSessionError: sessionがRUNNINGでない場合、または無限Sourceを含む場合。
        """

        runtime = self._running_runtime("flush")
        try:
            return runtime.flush()
        except PlanSessionError:
            raise
        except Exception as error:
            self._fail_and_finish(runtime, error)
            raise

    def close(self) -> RunResult:
        """Source受付を停止してpending入力をdrainし、resourceを解放する。

        Returns:
            resource解放直前の最終RunResult。
        """

        runtime = self._running_runtime("close")
        try:
            result = runtime.close()
            runtime.finish()
        except Exception as error:
            self._fail_and_finish(runtime, error)
            raise
        self._state = PlanSessionState.CLOSED
        return result

    def cancel(self) -> RunResult:
        """pending値をflushせずsessionを打ち切り、resourceを解放する。

        Returns:
            `SESSION_CANCELLED` Diagnosticを含む累積RunResult。
        """

        runtime = self._running_runtime("cancel")
        try:
            result = runtime.cancel()
            runtime.finish()
        except Exception as error:
            self._fail_and_finish(runtime, error)
            raise
        self._state = PlanSessionState.CANCELLED
        return result

    def _running_runtime(self, operation: str) -> _PlanRuntime:
        if self._state is not PlanSessionState.RUNNING or self._runtime is None:
            raise PlanSessionError(
                f"PlanSession.{operation} requires state=running; actual={self._state.value}"
            )
        return self._runtime

    def _fail_and_finish(self, runtime: _PlanRuntime, error: Exception) -> None:
        """元の失敗を保持したまま全resourceの解放を試みる。"""

        self._state = PlanSessionState.FAILED
        try:
            runtime.finish()
        except Exception as cleanup_error:
            error.add_note(f"PlanSession resource cleanup also failed: {cleanup_error}")

    @staticmethod
    def _logical_time_fraction(value: LogicalTime | Fraction | int | float) -> Fraction:
        if isinstance(value, LogicalTime):
            return value.as_fraction()
        try:
            return Fraction(str(value)) if isinstance(value, float) else Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise PlanSessionError("logical_end must be a finite rational value") from error


class _PlanRuntime:
    def __init__(
        self,
        plan: ExecutionPlan,
        duration: float | None,
        extensions: tuple[_BoundExtension, ...],
        *,
        continuous: bool = False,
        options: RuntimeOptions,
    ) -> None:
        self.plan = plan
        self.duration = None if duration is None else Fraction(str(duration))
        self.continuous = continuous
        self.options = options
        self.nodes_by_port = {
            output_port: node for node in plan._nodes for output_port in node.output_ports
        }
        self.nodes_by_id = {node.id: node for node in plan._nodes}
        signatures = _time_signature(plan._nodes)
        self.node_max_items = {node.id: _node_max_items(node, signatures) for node in plan._nodes}
        buffer_descriptors = {item.buffer_id: item for item in plan._portable_ir.buffers}
        port_buffer_descriptors = {
            item.producer_port_id: item
            for item in plan._portable_ir.buffers
            if item.kind == "port_shared"
        }
        self.port_buffers: dict[int, PortBuffer[Emission[object]]] = {
            output_port: PortBuffer(
                output_port,
                max_items=self._runtime_buffer_capacity(
                    output_port,
                    port_buffer_descriptors[output_port].max_items,
                    options.port_high_watermark,
                ),
            )
            for node in plan._nodes
            for output_port in node.output_ports
        }
        self.queues: dict[tuple[int, int], CursorQueue[Emission[object]]] = {}
        cursor_id = 0
        for node in plan._nodes:
            for input_index, item in enumerate(node.inputs):
                buffer = self.port_buffers[item.source_port]
                buffer.register_consumer(cursor_id)
                self.queues[(node.id, input_index)] = CursorQueue(buffer, cursor_id)
                cursor_id += 1
        self.latest: dict[tuple[int, int], LatestStateBuffer[Emission[object]]] = {}
        self.flexible_current: dict[tuple[int, int], Emission[object]] = {}
        self.frames: dict[int, _FrameState] = {}
        for descriptor in buffer_descriptors.values():
            owner = (descriptor.owner_node_id, descriptor.owner_input_index)
            if descriptor.kind == "frame_history":
                if descriptor.owner_node_id is None:
                    raise RuntimeError(
                        f"buffer {descriptor.buffer_id} FRAME_HISTORY lacks owner node"
                    )
                capacity = self._required_buffer_capacity(
                    descriptor.producer_port_id,
                    descriptor.max_items,
                )
                self.frames[descriptor.owner_node_id] = _FrameState(
                    FrameHistoryBuffer(descriptor.buffer_id, capacity)
                )
            elif descriptor.kind == "latest_state":
                if owner[0] is None or owner[1] is None:
                    raise RuntimeError(
                        f"buffer {descriptor.buffer_id} LATEST_STATE lacks owner input"
                    )
                if descriptor.max_items != 1:
                    raise RuntimeError(
                        f"buffer {descriptor.buffer_id} LATEST_STATE requires max_items=1"
                    )
                self.latest[(owner[0], owner[1])] = LatestStateBuffer(descriptor.buffer_id)
        self.rates: dict[int, _RateState] = {}
        self.port_sequences: dict[int, int] = defaultdict(int)
        self.published_counts: Counter[int] = Counter()
        self.port_frontiers: dict[int, Fraction] = {}
        self.port_gaps: dict[int, list[LogicalInterval]] = defaultdict(list)
        self.exhausted_ports: set[int] = set()
        self.stalled_nodes: set[int] = set()
        self.source_indexes: dict[int, int] = defaultdict(int)
        self.source_pending: dict[int, Emission[object]] = {}
        self.boundary_blocked_ports: set[int] = set()
        source_nodes = tuple(node for node in plan._nodes if node.kind is NodeKind.SOURCE)
        self.source_iterators = {
            node.id: iter(self._source_values(node))
            for node in source_nodes
            if not isinstance(node.source, RealtimeSource)
        }
        self.realtime_ingresses: dict[int, RealtimeIngressBuffer[object]] = {}
        for descriptor in buffer_descriptors.values():
            if descriptor.kind != "realtime_ingress":
                continue
            if descriptor.owner_node_id is None or descriptor.max_items is None:
                raise RuntimeError(
                    f"buffer {descriptor.buffer_id} REALTIME_INGRESS lacks owner or capacity"
                )
            source = self.nodes_by_id[descriptor.owner_node_id].source
            if not isinstance(source, RealtimeSource):
                raise RuntimeError(
                    f"buffer {descriptor.buffer_id} owner node {descriptor.owner_node_id} "
                    "is not a RealtimeSource"
                )
            self.realtime_ingresses[descriptor.owner_node_id] = RealtimeIngressBuffer(
                descriptor.buffer_id,
                descriptor.owner_node_id,
                descriptor.producer_port_id,
                descriptor.max_items,
                source.overflow_policy,
            )
        self.realtime_sessions: dict[int, RealtimeSourceSession] = {}
        self.pending_source_gaps: dict[int, list[Diagnostic]] = defaultdict(list)
        self.diagnostics = list(plan._compile_diagnostics)
        self.status_counts: Counter[EmissionStatus] = Counter()
        self.collectors: dict[int, CollectorSession[Any]] = {
            item.flow.port_id: item.collector.create_session() for item in plan._outputs
        }
        self.kernel_sessions: dict[int, CompiledKernelSession[object]] = {
            node_id: kernel.create_session() for node_id, kernel in plan._compiled_kernels.items()
        }
        self.extensions = tuple(self._create_extension_runtime(item) for item in extensions)
        self.output_ports = set(self.collectors)
        self.root_ports: set[int] = set()
        self._context = PlanContext(required_node_count=len(self.plan._nodes))
        self._initialized_extensions: list[_ExtensionRuntime] = []
        self._roots: tuple[int, ...] = ()
        self._seen_counts: Counter[int] = Counter()
        self._active: set[int] = set()
        self._started = False
        self._finished = False
        self._cancelled = False
        self._reported_unmatched = False
        self.last_budget_exhausted = False
        self._scheduler_steps = 0
        self._kernel_calls: Counter[int] = Counter()
        self._kernel_total_ns: Counter[int] = Counter()
        self._kernel_max_ns: Counter[int] = Counter()

    @staticmethod
    def _required_buffer_capacity(port_id: int, max_items: int | None) -> int:
        """PortablePlanIRの通常Port capacityが実行可能な正値であることを検証する。"""

        if max_items is None or max_items <= 0:
            raise RuntimeError(
                f"port {port_id} PORT_SHARED buffer lacks a positive max_items contract"
            )
        return max_items

    @classmethod
    def _runtime_buffer_capacity(
        cls,
        port_id: int,
        compiled: int | None,
        requested: int | None,
    ) -> int:
        """RuntimeOptions watermarkがcompile済み下限を満たすことを検証する。"""

        required = cls._required_buffer_capacity(port_id, compiled)
        if requested is not None and requested < required:
            raise PlanSessionError(
                f"port {port_id} requested high_watermark={requested} is below "
                f"compiled capacity={required}; contract=bounded_shared_buffer"
            )
        return required if requested is None else requested

    def _create_extension_runtime(self, bound: _BoundExtension) -> _ExtensionRuntime:
        """binding factoryから型検証済みのrun-local handlerを生成する。"""

        session = bound.binding.create_session()
        if not isinstance(session, ExtensionSession):
            item = bound.observation
            node = self.plan._graph.node_for_port(item.flow.port_id)
            raise ExtensionBindingError(
                f"extension_id {item.extension_id!r} slot 'extension:{item.extension_id}' "
                f"node {node.id} port {item.flow.port_id} create_session returned an "
                "invalid handler; contract=ExtensionSession"
            )
        return _ExtensionRuntime(
            bound.observation,
            session,
            bound.observation.trigger.create_session(),
        )

    def run(self) -> RunResult:
        """v0.1互換の一回実行をlifecycle primitiveで完了する。"""

        try:
            self.start()
            self._drive()
            if not self._active:
                self._report_unmatched_once()
            return self._result()
        finally:
            self.finish()

    def start(self) -> None:
        """ExtensionとRealtime Sourceを開始し、root需要状態を初期化する。"""

        if self._started:
            raise RuntimeError("Plan runtime is already started")
        for extension in self.extensions:
            self._initialize_extension(extension, self._context)
            self._initialized_extensions.append(extension)
        self._start_realtime_sources()
        self._roots = tuple(
            dict.fromkeys(
                [item.flow.port_id for item in self.plan._outputs]
                + [extension.observation.flow.port_id for extension in self.extensions]
            )
        )
        self.root_ports = set(self._roots)
        self._active = set(self._roots)
        self._started = True
        if self.continuous:
            diagnostic = Diagnostic(
                Severity.INFO,
                "SESSION_STARTED",
                "PlanSession acquired run-local resources",
                details={
                    "kernel_sessions": sorted(self.kernel_sessions),
                    "realtime_source_sessions": sorted(self.realtime_sessions),
                    "buffer_ids": sorted(buffer.buffer_id for buffer in self.port_buffers.values()),
                },
            )
            self.diagnostics.append(diagnostic)
            self._notify_diagnostic(diagnostic)

    def run_until(self, logical_end: Fraction) -> RunResult:
        """状態を保持したまま排他的な論理時間上限まで進める。"""

        self.duration = logical_end
        self.boundary_blocked_ports.clear()
        self._drive()
        if not self._active:
            self._report_unmatched_once()
        return self._result()

    def flush(self) -> RunResult:
        """finiteまたは既にclose済みSourceをdrainする。"""

        open_realtime = [
            node_id for node_id, ingress in self.realtime_ingresses.items() if not ingress.is_closed
        ]
        non_finite_pull = [
            item.node_id
            for item in self.plan._portable_ir.sources
            if not item.is_finite and item.mode != "realtime_push"
        ]
        if open_realtime or non_finite_pull:
            raise PlanSessionError(
                "PlanSession.flush requires finite or already closed Sources; "
                f"open_realtime_nodes={open_realtime}; non_finite_pull_nodes={non_finite_pull}"
            )
        self.duration = None
        self.boundary_blocked_ports.clear()
        self._drive()
        if not self._active:
            self._report_unmatched_once()
        return self._result()

    def close(self) -> RunResult:
        """Realtime受付を停止・closeして全Sourceと下流状態をdrainする。"""

        self._stop_realtime_sources(close_ingress=True)
        non_finite_pull = [
            item.node_id
            for item in self.plan._portable_ir.sources
            if not item.is_finite and item.mode != "realtime_push"
        ]
        if non_finite_pull:
            raise PlanSessionError(
                "PlanSession.close cannot infer EOF for non-finite pull Sources; "
                f"nodes={non_finite_pull}; call cancel()"
            )
        self.duration = None
        self.boundary_blocked_ports.clear()
        self._drive()
        if not self._active:
            self._report_unmatched_once()
        diagnostic = Diagnostic(
            Severity.INFO,
            "SESSION_CLOSED",
            "PlanSession stopped Sources and drained run-local resources",
            details={
                "remaining_active_ports": sorted(self._active),
                "realtime_pending_items": {
                    node_id: ingress.pending_count
                    for node_id, ingress in self.realtime_ingresses.items()
                },
            },
        )
        self.diagnostics.append(diagnostic)
        self._notify_diagnostic(diagnostic)
        return self._result()

    def cancel(self) -> RunResult:
        """pending処理を破棄したことをDiagnosticへ記録する。"""

        self._stop_realtime_sources(close_ingress=False)
        discarded = {
            node_id: ingress.discard() for node_id, ingress in self.realtime_ingresses.items()
        }
        diagnostic = Diagnostic(
            Severity.WARNING,
            "SESSION_CANCELLED",
            "PlanSession was cancelled without flushing pending state",
            details={
                "active_ports": sorted(self._active),
                "discarded_realtime_items": discarded,
            },
        )
        self.diagnostics.append(diagnostic)
        self._notify_diagnostic(diagnostic)
        self._cancelled = True
        return self._result()

    def finish(self) -> None:
        """run-local SourceとExtension resourceを一度だけ解放する。"""

        if self._finished:
            return
        self._finished = True
        first_error: Exception | None = None
        try:
            self._stop_realtime_sources(close_ingress=True)
        except Exception as error:
            first_error = error
        for extension in reversed(self._initialized_extensions):
            try:
                self._finalize_extension(extension, self._context)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _drive(self) -> None:
        """active rootを境界、EOF、stallのいずれかまで決定的に進める。"""

        steps = 0
        self.last_budget_exhausted = False
        while self._active:
            progressed = False
            for port_id in self._roots:
                if port_id not in self._active:
                    continue
                if self._seen_counts[port_id] < self.published_counts[port_id]:
                    self._seen_counts[port_id] += 1
                    progressed = True
                    continue
                if port_id in self.exhausted_ports:
                    self._active.remove(port_id)
                    continue
                if self._advance_port(port_id):
                    progressed = True
                    steps += 1
                    self._scheduler_steps += 1
                elif port_id in self.exhausted_ports:
                    self._active.remove(port_id)
            if (
                self._active
                and self.options.max_scheduler_steps is not None
                and steps >= self.options.max_scheduler_steps
            ):
                self.last_budget_exhausted = True
                diagnostic = Diagnostic(
                    Severity.INFO,
                    "EXECUTION_BUDGET_EXHAUSTED",
                    "scheduler step budget ended this execution slice",
                    details={"max_scheduler_steps": self.options.max_scheduler_steps},
                )
                self.diagnostics.append(diagnostic)
                self._notify_diagnostic(diagnostic)
                break
            if self._active and not progressed:
                if self.boundary_blocked_ports:
                    break
                self._report_scheduler_deadlock(self._active)
                break

    def _report_unmatched_once(self) -> None:
        if self._reported_unmatched:
            return
        self._report_unmatched_inputs()
        self._reported_unmatched = True

    def _initialize_extension(
        self,
        extension: _ExtensionRuntime,
        context: PlanContext,
    ) -> None:
        """handler初期化失敗へ観測契約の識別情報を付ける。"""

        try:
            extension.session.initialize(context)
        except Exception as error:
            raise self._extension_execution_error(extension, "initialize", error) from error

    def _start_realtime_sources(self) -> None:
        """run-local ingressをbindして外部push受付を開始する。"""

        for node_id, ingress in self.realtime_ingresses.items():
            node = self.nodes_by_id[node_id]
            source = node.source
            if not isinstance(source, RealtimeSource):
                raise RuntimeError(f"realtime source node {node_id} lost its binding")
            try:
                session = source.start(ingress, node.config)
            except Exception as error:
                raise SourceExecutionError(
                    f"realtime source node {node.id} port {node.output_port} start failed: "
                    f"{error}; contract=RealtimeSource.start"
                ) from error
            if not isinstance(session, RealtimeSourceSession):
                raise TypeError(
                    f"realtime source node {node.id} port {node.output_port} start returned "
                    "an invalid RealtimeSourceSession"
                )
            self.realtime_sessions[node_id] = session

    def _stop_realtime_sources(self, *, close_ingress: bool) -> None:
        """各Realtime Sourceを一度だけ停止し、必要ならingress受付も閉じる。"""

        first_error: SourceExecutionError | None = None
        for node_id, session in tuple(self.realtime_sessions.items()):
            node = self.nodes_by_id[node_id]
            try:
                session.stop()
            except Exception as error:
                if first_error is None:
                    first_error = SourceExecutionError(
                        f"realtime source node {node.id} port {node.output_port} stop failed: "
                        f"{error}; contract=RealtimeSourceSession.stop"
                    )
            finally:
                self.realtime_sessions.pop(node_id, None)
        if close_ingress:
            for ingress in self.realtime_ingresses.values():
                ingress.close()
        if first_error is not None:
            raise first_error

    def _finalize_extension(
        self,
        extension: _ExtensionRuntime,
        context: PlanContext,
    ) -> None:
        """handler終了失敗へ観測契約の識別情報を付ける。"""

        try:
            extension.session.finalize(context)
        except Exception as error:
            raise self._extension_execution_error(extension, "finalize", error) from error

    def _extension_execution_error(
        self,
        extension: _ExtensionRuntime,
        callback: str,
        error: Exception,
    ) -> ExtensionExecutionError:
        item = extension.observation
        node = self.plan._graph.node_for_port(item.flow.port_id)
        return ExtensionExecutionError(
            f"extension_id {item.extension_id!r} slot 'extension:{item.extension_id}' "
            f"node {node.id} port {item.flow.port_id} callback {callback!r} failed: {error}; "
            f"contract=failure_policy:{item.failure_policy.value}"
        )

    def _notify_diagnostic(self, diagnostic: Diagnostic) -> None:
        """runtime Diagnosticをpriority順に全handlerへ配送する。"""

        for extension in self.extensions:
            try:
                extension.session.on_diagnostic(diagnostic)
            except Exception as error:
                raise self._extension_execution_error(extension, "on_diagnostic", error) from error

    @staticmethod
    def _source_emission(value: object, index: int) -> Emission[object]:
        if isinstance(value, Emission):
            return value
        return Emission(
            value=value,
            interval=LogicalInterval(LogicalTime(index), LogicalTime(index + 1)),
            sequence=index,
        )

    def _source_values(self, node: NodeSpec) -> Iterator[object]:
        source = node.source
        if source is None:
            return
        if isinstance(source, RealtimeSource):
            raise RuntimeError("RealtimeSource must be consumed through REALTIME_INGRESS")
        if not isinstance(source, Source):
            yield from source
            return
        if not source.is_finite and self.duration is None and not self.continuous:
            raise ValueError("generated Source requires run(duration=...)")

        logical_start = LogicalTime(0)
        while True:
            if self.duration is not None and logical_start.as_fraction() >= self.duration:
                if self.continuous:
                    yield _SOURCE_BOUNDARY
                    continue
                return
            request_duration = (
                self.options.source_chunk_duration
                or self.plan._source_request_periods.get(node.id, Fraction(1))
            )
            batch = source.read(SourceRequest(logical_start, request_duration), node.config)
            if not batch.emissions and not batch.eof:
                raise RuntimeError("Source returned an empty non-EOF batch")
            yield from batch.emissions
            if batch.eof:
                return
            if batch.emissions:
                logical_start = batch.emissions[-1].interval.end
            else:
                return

    def _publish(self, port_id: int, emission: Emission[object]) -> None:
        self.status_counts[emission.status] += 1
        event = OutputEvent(port_id, emission)
        # 観測可能な劣化結果をcollector overflowより先に保存できる順序を正本とする。
        for extension in self.extensions:
            if extension.observation.flow.port_id != port_id:
                continue
            if not extension.trigger.should_fire(emission):
                continue
            try:
                extension.session.on_output(event)
            except Exception as error:
                raise self._extension_execution_error(extension, "on_output", error) from error
        if port_id in self.output_ports:
            self.collectors[port_id].add(emission)
        self.port_buffers[port_id].publish(emission)
        self.published_counts[port_id] += 1

    def _advance_frontier(self, port_id: int, value: Fraction) -> None:
        """Portが今後生成し得ない過去区間の終端を単調に進める。"""

        current = self.port_frontiers.get(port_id)
        if current is None or value > current:
            self.port_frontiers[port_id] = value

    def _advance_port(self, port_id: int) -> bool:
        """portの未充足需要へ向けて高々一つの実行単位だけを進める。"""

        if port_id in self.exhausted_ports:
            return False
        node = self.nodes_by_port[port_id]
        demanded_outputs = [
            output_port
            for output_port in node.output_ports
            if output_port in self.root_ports or self.port_buffers[output_port].consumer_count > 0
        ]
        if any(
            not self.port_buffers[output_port].can_publish(self.node_max_items[node.id])
            for output_port in demanded_outputs
        ):
            return False
        if node.kind is NodeKind.SOURCE:
            return self._advance_source(node)
        if node.kind is NodeKind.FRAME:
            return self._advance_frame(node)
        if node.kind is NodeKind.RATE:
            return self._advance_rate(node)
        if node.kind is NodeKind.MAP:
            return self._advance_map(node)
        raise RuntimeError(f"unsupported Node kind {node.kind!r}")

    def _advance_source(self, node: NodeSpec) -> bool:
        """需要があるSourceから一件だけpublishする。"""

        pending = self.source_pending.get(node.id)
        if pending is not None:
            return self._publish_source_at_boundary(node, pending)

        ingress = self.realtime_ingresses.get(node.id)
        if ingress is not None:
            try:
                record = ingress.take()
            except Exception as error:
                raise SourceExecutionError(
                    f"realtime source node {node.id} port {node.output_port} receive failed: "
                    f"{error}; contract=RealtimeReceiver.fail"
                ) from error
            if record is None:
                self.exhausted_ports.add(node.output_port)
                return True
            if isinstance(record, GapMarker):
                self._record_source_gap(node, record)
                return True
            emission = self._degrade_after_source_gap(node, record)
            return self._publish_source_at_boundary(node, emission)

        try:
            value = next(self.source_iterators[node.id])
        except StopIteration:
            self.exhausted_ports.add(node.output_port)
            return True
        if value is _SOURCE_BOUNDARY:
            self.boundary_blocked_ports.add(node.output_port)
            return False
        emission = self._source_emission(value, self.source_indexes[node.id])
        self.source_indexes[node.id] += 1
        return self._publish_source_at_boundary(node, emission)

    def _publish_source_at_boundary(
        self,
        node: NodeSpec,
        emission: Emission[object],
    ) -> bool:
        """境界外Emissionを失わず次のrun_untilまで保留する。"""

        if self.duration is not None and emission.interval.end.as_fraction() > self.duration:
            self.source_pending[node.id] = emission
            self.boundary_blocked_ports.add(node.output_port)
            return False
        self.source_pending.pop(node.id, None)
        self.boundary_blocked_ports.discard(node.output_port)
        self._publish(node.output_port, emission)
        self._advance_frontier(node.output_port, emission.interval.end.as_fraction())
        return True

    def _record_source_gap(self, node: NodeSpec, marker: GapMarker) -> None:
        """dropを失われないDiagnosticとして記録し、次Emissionへ引き継ぐ。"""

        diagnostic = Diagnostic(
            Severity.WARNING,
            "INPUT_OVERRUN",
            "realtime ingress dropped input emissions",
            node_id=node.id,
            port_id=node.output_port,
            interval=marker.interval,
            details={
                "dropped_count": marker.dropped_count,
                "total_dropped_count": marker.total_dropped_count,
                "capacity": marker.capacity,
                "overflow_policy": marker.overflow_policy.value,
            },
        )
        self.pending_source_gaps[node.id].append(diagnostic)
        self.diagnostics.append(diagnostic)
        self._record_port_gap(node.output_port, marker.interval)
        self._advance_frontier(node.output_port, marker.interval.end.as_fraction())
        self._notify_diagnostic(diagnostic)

    def _degrade_after_source_gap(
        self,
        node: NodeSpec,
        emission: Emission[object],
    ) -> Emission[object]:
        """直前のGapMarker群を次の受理Emissionへ付加する。"""

        diagnostics = tuple(self.pending_source_gaps.pop(node.id, ()))
        if not diagnostics:
            return emission
        metadata = dict(emission.metadata)
        dropped_count = 0
        for diagnostic in diagnostics:
            value = diagnostic.details.get("dropped_count")
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("INPUT_OVERRUN Diagnostic lacks integer dropped_count")
            dropped_count += value
        metadata["input_overrun_dropped_count"] = dropped_count
        return Emission(
            emission.value,
            emission.interval,
            emission.sequence,
            max((emission.status, EmissionStatus.DEGRADED), key=_status_rank),
            emission.diagnostics + diagnostics,
            metadata,
        )

    def _record_port_gap(self, port_id: int, interval: LogicalInterval) -> None:
        """Portで生成不能なgap intervalをrun-localに順序保持する。"""

        gaps = self.port_gaps[port_id]
        if gaps and gaps[-1].end == interval.start:
            gaps[-1] = LogicalInterval(gaps[-1].start, interval.end)
            return
        if interval not in gaps:
            gaps.append(interval)

    def _record_emission_gaps(
        self,
        port_id: int,
        emissions: Sequence[Emission[object]],
    ) -> None:
        """入力Diagnosticが示すgapを変換後Portのcontrol stateへ伝播する。"""

        for emission in emissions:
            for diagnostic in emission.diagnostics:
                if diagnostic.code == "INPUT_OVERRUN" and diagnostic.interval is not None:
                    self._record_port_gap(port_id, diagnostic.interval)

    def _gap_for_interval(
        self,
        port_id: int,
        required: LogicalInterval,
    ) -> LogicalInterval | None:
        """required intervalと重なる既知gapを返す。"""

        required_start = required.start.as_fraction()
        required_end = required.end.as_fraction()
        for gap in self.port_gaps.get(port_id, ()):
            if gap.start.as_fraction() < required_end and required_start < gap.end.as_fraction():
                return gap
        return None

    def _advance_frame(self, node: NodeSpec) -> bool:
        """frame履歴を一件進め、入力EOFでは必要なら一度だけpaddingする。"""

        queue = self.queues[(node.id, 0)]
        if queue:
            return self._process_frame_input(node)
        input_port = node.inputs[0].source_port
        if input_port not in self.exhausted_ports:
            return self._advance_port(input_port)
        if self._flush_padded_frame(node):
            return True
        self.exhausted_ports.add(node.output_port)
        return True

    def _advance_rate(self, node: NodeSpec) -> bool:
        """RATE入力を一件処理し、未到着ならそのproducerだけを進める。"""

        queue = self.queues[(node.id, 0)]
        if queue:
            return self._process_rate_input(node)
        input_port = node.inputs[0].source_port
        if input_port in self.exhausted_ports:
            self.exhausted_ports.add(node.output_port)
            return True
        return self._advance_port(input_port)

    def _advance_map(self, node: NodeSpec) -> bool:
        """MAPの次intervalに必要な不足入力だけを一段進める。"""

        if node.id in self.stalled_nodes:
            return False
        main_queue = self.queues[(node.id, 0)]
        main_port = node.inputs[0].source_port
        if not main_queue:
            for input_index, input_spec in enumerate(node.inputs[1:], start=1):
                if input_spec.semantics in {
                    InputSemantics.SYNCHRONOUS,
                    InputSemantics.LATEST,
                }:
                    continue
                queue = self.queues[(node.id, input_index)]
                if (
                    not queue
                    and input_spec.source_port not in self.exhausted_ports
                    and self._advance_port(input_spec.source_port)
                ):
                    return True
            if main_port in self.exhausted_ports:
                self.exhausted_ports.update(node.output_ports)
                return True
            return self._advance_port(main_port)
        main = main_queue[0]

        latest_progress = self._advance_missing_latest(node, main)
        if latest_progress is not None:
            return latest_progress

        for input_index, input_spec in enumerate(node.inputs[1:], start=1):
            if input_spec.semantics in {
                InputSemantics.SYNCHRONOUS,
                InputSemantics.LATEST,
            }:
                continue
            flexible_progress = self._advance_flexible_input(node, input_index, input_spec, main)
            if flexible_progress is not None:
                return flexible_progress

        for input_index, input_spec in enumerate(node.inputs[1:], start=1):
            if input_spec.semantics is not InputSemantics.SYNCHRONOUS:
                continue
            queue = self.queues[(node.id, input_index)]
            if not queue:
                if input_spec.source_port in self.exhausted_ports:
                    gap = self._gap_for_interval(input_spec.source_port, main.interval)
                    if gap is not None:
                        self._skip_exact_merge_gap(node, input_index, main.interval, gap)
                        return True
                    self._stall_exact_merge(node, input_index, main.interval, None, None)
                    return True
                frontier = self.port_frontiers.get(input_spec.source_port)
                if frontier is not None and frontier > main.interval.start.as_fraction():
                    gap = self._gap_for_interval(input_spec.source_port, main.interval)
                    if gap is not None:
                        self._skip_exact_merge_gap(node, input_index, main.interval, gap)
                        return True
                    self._stall_exact_merge(
                        node,
                        input_index,
                        main.interval,
                        None,
                        frontier,
                    )
                    return True
                return self._advance_port(input_spec.source_port)
            available = queue[0]
            if available.interval == main.interval:
                continue
            main_start = main.interval.start.as_fraction()
            available_start = available.interval.start.as_fraction()
            if available_start < main_start:
                queue.popleft()
                return True
            gap = self._gap_for_interval(input_spec.source_port, main.interval)
            if gap is not None:
                self._skip_exact_merge_gap(node, input_index, main.interval, gap)
                return True
            self._stall_exact_merge(
                node,
                input_index,
                main.interval,
                available.interval,
                self.port_frontiers.get(input_spec.source_port),
            )
            return True
        return self._process_map_if_ready(node)

    def _advance_flexible_input(
        self,
        node: NodeSpec,
        input_index: int,
        input_spec: InputSpec,
        main: Emission[object],
    ) -> bool | None:
        """包含、overlap、tolerance入力をsequence最小の候補へ合わせる。"""

        queue = self.queues[(node.id, input_index)]
        current = self.flexible_current.get((node.id, input_index))
        if current is not None:
            if self._flexible_match(input_spec, main.interval, current.interval):
                return None
            if self._candidate_is_before(input_spec, main.interval, current.interval):
                self.flexible_current.pop((node.id, input_index), None)
                return True
            return self._handle_missing_flexible(node, input_index, input_spec, main.interval)
        if queue:
            candidate = queue[0]
            if self._flexible_match(input_spec, main.interval, candidate.interval):
                self.flexible_current[(node.id, input_index)] = queue.popleft()
                return None
            if self._candidate_is_before(input_spec, main.interval, candidate.interval):
                queue.popleft()
                return True
            return self._handle_missing_flexible(node, input_index, input_spec, main.interval)
        if input_spec.source_port not in self.exhausted_ports:
            return self._advance_port(input_spec.source_port)
        return self._handle_missing_flexible(node, input_index, input_spec, main.interval)

    @staticmethod
    def _flexible_match(
        input_spec: InputSpec,
        reference: LogicalInterval,
        candidate: LogicalInterval,
    ) -> bool:
        if input_spec.semantics is InputSemantics.CONTAINS:
            return candidate.start <= reference.start and candidate.end >= reference.end
        if input_spec.semantics is InputSemantics.OVERLAPS:
            return candidate.start < reference.end and candidate.end > reference.start
        if input_spec.semantics is InputSemantics.TOLERANCE:
            tolerance = input_spec.tolerance
            if tolerance is None:
                raise RuntimeError("tolerance input lacks a tolerance contract")
            return (
                abs(candidate.start.as_fraction() - reference.start.as_fraction()) <= tolerance
                and abs(candidate.end.as_fraction() - reference.end.as_fraction()) <= tolerance
            )
        raise RuntimeError(f"unsupported flexible synchronization {input_spec.semantics!r}")

    @staticmethod
    def _candidate_is_before(
        input_spec: InputSpec,
        reference: LogicalInterval,
        candidate: LogicalInterval,
    ) -> bool:
        if input_spec.semantics in {InputSemantics.CONTAINS, InputSemantics.OVERLAPS}:
            return candidate.end <= reference.start
        tolerance = input_spec.tolerance
        if tolerance is None:
            raise RuntimeError("tolerance input lacks a tolerance contract")
        return candidate.end.as_fraction() < reference.end.as_fraction() - tolerance

    def _handle_missing_flexible(
        self,
        node: NodeSpec,
        input_index: int,
        input_spec: InputSpec,
        required: LogicalInterval,
    ) -> bool:
        """生成不能なflexible同期を明示policyどおり停止またはskipする。"""

        diagnostic = Diagnostic(
            Severity.WARNING,
            (
                "SYNC_INPUT_SKIPPED"
                if input_spec.missing_policy is MissingInputPolicy.SKIP
                else "STALLED_SYNCHRONIZATION"
            ),
            "synchronized input cannot produce a matching interval",
            node_id=node.id,
            port_id=node.output_port,
            interval=required,
            details={
                "failed_input_index": input_index,
                "semantics": input_spec.semantics.value,
                "missing_policy": input_spec.missing_policy.value,
                "tolerance": (None if input_spec.tolerance is None else str(input_spec.tolerance)),
                "tie_break": "lowest_sequence",
                "reference_input_index": 0,
            },
        )
        if input_spec.missing_policy is MissingInputPolicy.STALL:
            self._stop_node(node, diagnostic)
            return True
        self.queues[(node.id, 0)].popleft()
        self.diagnostics.append(diagnostic)
        self._advance_frontier(node.output_port, required.end.as_fraction())
        self._notify_diagnostic(diagnostic)
        return True

    def _advance_missing_latest(
        self,
        node: NodeSpec,
        main: Emission[object],
    ) -> bool | None:
        """latest入力をmain時刻まで進め、未充足時だけ進捗結果を返す。"""

        for input_index, input_spec in enumerate(node.inputs):
            if input_spec.semantics is not InputSemantics.LATEST:
                continue
            queue = self.queues[(node.id, input_index)]
            consumed = False
            while queue and queue[0].interval.start <= main.interval.start:
                self.latest[(node.id, input_index)].replace(queue.popleft())
                consumed = True
            if consumed:
                return True
            if queue:
                if self.latest[(node.id, input_index)].has_value:
                    continue
                self._stall_latest(node, input_index, main.interval)
                return True
            frontier = self.port_frontiers.get(input_spec.source_port)
            if frontier is not None and frontier > main.interval.start.as_fraction():
                if self.latest[(node.id, input_index)].has_value:
                    continue
                self._stall_latest(node, input_index, main.interval)
                return True
            if input_spec.source_port not in self.exhausted_ports:
                return self._advance_port(input_spec.source_port)
            if not self.latest[(node.id, input_index)].has_value:
                self._stall_latest(node, input_index, main.interval)
                return True
        return None

    def _stall_exact_merge(
        self,
        node: NodeSpec,
        input_index: int,
        required: LogicalInterval,
        available: LogicalInterval | None,
        producer_frontier: Fraction | None,
    ) -> None:
        """生成不能なexact intervalを診断し、Nodeの全cursorを解除する。"""

        diagnostic = Diagnostic(
            Severity.WARNING,
            "STALLED_EXACT_MERGE",
            "exact merge input can no longer produce the required interval",
            node_id=node.id,
            port_id=node.output_port,
            interval=required,
            details={
                "input_ports": [item.source_port for item in node.inputs],
                "failed_input_index": input_index,
                "required_interval": str(required),
                "available_interval": None if available is None else str(available),
                "producer_frontier": (
                    None if producer_frontier is None else str(producer_frontier)
                ),
            },
        )
        self._stop_node(node, diagnostic)

    def _skip_exact_merge_gap(
        self,
        node: NodeSpec,
        input_index: int,
        required: LogicalInterval,
        gap: LogicalInterval,
    ) -> None:
        """GapMarkerで生成不能と証明された一intervalだけを解放する。"""

        for index, input_spec in enumerate(node.inputs):
            if input_spec.semantics is not InputSemantics.SYNCHRONOUS:
                continue
            queue = self.queues[(node.id, index)]
            if queue and queue[0].interval == required:
                queue.popleft()
        diagnostic = Diagnostic(
            Severity.WARNING,
            "MERGE_INPUT_GAP",
            "exact merge skipped an interval proven missing by realtime input gap",
            node_id=node.id,
            port_id=node.output_port,
            interval=required,
            details={
                "failed_input_index": input_index,
                "failed_port_id": node.inputs[input_index].source_port,
                "gap_interval": str(gap),
                "contract": "gap_resynchronize_at_common_frontier",
            },
        )
        self.diagnostics.append(diagnostic)
        self._record_port_gap(node.output_port, required)
        self._advance_frontier(node.output_port, required.end.as_fraction())
        self._notify_diagnostic(diagnostic)

    def _stall_latest(
        self,
        node: NodeSpec,
        input_index: int,
        required: LogicalInterval,
    ) -> None:
        """初期latest値が得られないNodeを診断して停止する。"""

        diagnostic = Diagnostic(
            Severity.WARNING,
            "STALLED_LATEST_INPUT",
            "latest input has no value at or before the required interval",
            node_id=node.id,
            port_id=node.output_port,
            interval=required,
            details={"failed_input_index": input_index},
        )
        self._stop_node(node, diagnostic)

    def _stop_node(self, node: NodeSpec, diagnostic: Diagnostic) -> None:
        """停止Nodeのcursorを解除し、無関係な観測経路を継続可能にする。"""

        self.stalled_nodes.add(node.id)
        self.exhausted_ports.update(node.output_ports)
        self.diagnostics.append(diagnostic)
        for input_index, _ in enumerate(node.inputs):
            self.queues[(node.id, input_index)].close()
        for input_spec in node.inputs:
            self._release_if_undemanded(input_spec.source_port)
        self._notify_diagnostic(diagnostic)

    def _release_if_undemanded(self, port_id: int) -> None:
        """終端にもconsumerにも到達しない経路のcursorを上流へ再帰的に解除する。"""

        if port_id in self.root_ports or self.port_buffers[port_id].consumer_count:
            return
        node = self.nodes_by_port[port_id]
        self.exhausted_ports.add(port_id)
        for input_index, input_spec in enumerate(node.inputs):
            self.queues[(node.id, input_index)].close()
            self._release_if_undemanded(input_spec.source_port)

    def _report_scheduler_deadlock(self, active_ports: set[int]) -> None:
        """容量待ちだけが残った実装不整合をruntime Diagnosticへ残す。"""

        diagnostic = Diagnostic(
            Severity.ERROR,
            "SCHEDULER_DEADLOCK",
            "no demanded path can advance within planned buffer capacities",
            details={"active_ports": sorted(active_ports)},
        )
        self.diagnostics.append(diagnostic)
        self._notify_diagnostic(diagnostic)

    def _process_frame_input(self, node: NodeSpec) -> bool:
        queue = self.queues[(node.id, 0)]
        if not queue:
            return False
        state = self.frames[node.id]
        emission = queue.popleft()
        self._record_emission_gaps(node.output_port, (emission,))
        if _has_input_overrun(emission):
            state.history.clear()
            state.skip_remaining = 0
        if state.skip_remaining:
            state.skip_remaining -= 1
            self._advance_frontier(node.output_port, emission.interval.end.as_fraction())
            return True
        state.history.append(emission)
        if node.frame_size is None or node.frame_hop is None:
            raise RuntimeError("FRAME Node lacks size or hop")
        if len(state.history) < node.frame_size:
            first = state.history.first
            if first is None:
                raise RuntimeError(f"FRAME Node {node.id} lost non-empty history")
            self._advance_frontier(
                node.output_port,
                first.interval.start.as_fraction(),
            )
            return True
        self._emit_frame(node, state.history.snapshot(node.frame_size))
        if node.frame_hop <= len(state.history):
            state.history.discard_prefix(node.frame_hop)
        else:
            state.history.clear()
            state.skip_remaining = node.frame_hop - node.frame_size
        first = state.history.first
        next_start = (
            first.interval.start.as_fraction()
            if first is not None
            else emission.interval.end.as_fraction()
        )
        self._advance_frontier(node.output_port, next_start)
        return True

    def _emit_frame(self, node: NodeSpec, items: Sequence[Emission[object]]) -> None:
        diagnostics = tuple(item for emission in items for item in emission.diagnostics)
        sequence = self.port_sequences[node.output_port]
        self.port_sequences[node.output_port] += 1
        frame = Emission(
            value=tuple(item.value for item in items),
            interval=LogicalInterval(items[0].interval.start, items[-1].interval.end),
            sequence=sequence,
            status=_combined_status(items),
            diagnostics=diagnostics,
        )
        self._publish(node.output_port, frame)

    def _flush_padded_frame(self, node: NodeSpec) -> bool:
        """有限入力EOFの未完成frameを一度だけpaddingしてpublishする。"""

        if not node.pad_end:
            return False
        state = self.frames.get(node.id)
        if state is None or not len(state.history) or node.frame_size is None:
            return False
        items = list(state.history.snapshot())
        last = items[-1]
        padding_count = node.frame_size - len(items)
        item_duration = last.interval.end.as_fraction() - last.interval.start.as_fraction()
        for offset in range(padding_count):
            start = last.interval.end.as_fraction() + offset * item_duration
            items.append(
                Emission(
                    None,
                    LogicalInterval(
                        self._time_from_fraction(start),
                        self._time_from_fraction(start + item_duration),
                    ),
                    last.sequence + offset + 1,
                    EmissionStatus.DEGRADED,
                    (
                        Diagnostic(
                            Severity.WARNING,
                            "FRAME_PADDED_AT_EOF",
                            "frame was padded at finite Source EOF",
                            node_id=node.id,
                        ),
                    ),
                )
            )
        self._emit_frame(node, items)
        state.history.clear()
        self._advance_frontier(node.output_port, items[-1].interval.end.as_fraction())
        return True

    @staticmethod
    def _time_from_fraction(value: Fraction) -> LogicalTime:
        """正確な有理時刻をLogicalTimeへ変換する。"""

        return LogicalTime(value.numerator, 1, value.denominator)

    def _process_rate_input(self, node: NodeSpec) -> bool:
        """入力interval内の発火境界へHOLD値を割り当てる。"""

        queue = self.queues[(node.id, 0)]
        if not queue:
            return False
        source = queue.popleft()
        self._record_emission_gaps(node.output_port, (source,))
        period = node.rate_period
        if period is None or node.rate_policy is None:
            raise RuntimeError("RATE Node lacks a supported period and policy")
        start = source.interval.start.as_fraction()
        end = source.interval.end.as_fraction()
        if node.rate_policy is RatePolicy.SAMPLE:
            if (start / period).denominator == 1:
                self._publish(
                    node.output_port,
                    Emission(
                        source.value,
                        source.interval,
                        self._next_sequence(node.output_port),
                        source.status,
                        source.diagnostics,
                        source.metadata,
                    ),
                )
            self._advance_frontier(node.output_port, end)
            return True
        if node.rate_policy is not RatePolicy.HOLD:
            raise RuntimeError("RATE Node has an unsupported policy")
        state = self.rates.setdefault(node.id, _RateState())
        if _has_input_overrun(source):
            state.next_fire = None
        if state.next_fire is None:
            state.next_fire = start
        while state.next_fire < start:
            skipped = -(-(start - state.next_fire) // period)
            state.next_fire += skipped * period
        while state.next_fire < end:
            fire = state.next_fire
            output = Emission(
                source.value,
                LogicalInterval(
                    self._time_from_fraction(fire),
                    self._time_from_fraction(fire + period),
                ),
                self._next_sequence(node.output_port),
                source.status,
                source.diagnostics,
                source.metadata,
            )
            self._publish(node.output_port, output)
            state.next_fire += period
        self._advance_frontier(node.output_port, end)
        return True

    def _prepare_latest(self, node: NodeSpec, main: Emission[object]) -> bool:
        for index, input_spec in enumerate(node.inputs):
            if input_spec.semantics is not InputSemantics.LATEST:
                continue
            queue = self.queues[(node.id, index)]
            while queue and queue[0].interval.start <= main.interval.start:
                self.latest[(node.id, index)].replace(queue.popleft())
            if not self.latest[(node.id, index)].has_value:
                return False
        return True

    def _process_map_if_ready(self, node: NodeSpec) -> bool:
        main_queue = self.queues[(node.id, 0)]
        if not main_queue:
            return False
        main = main_queue[0]
        if not self._prepare_latest(node, main):
            return False
        for index, input_spec in enumerate(node.inputs[1:], start=1):
            if input_spec.semantics is InputSemantics.SYNCHRONOUS:
                queue = self.queues[(node.id, index)]
                if not queue or queue[0].interval != main.interval:
                    return False

        inputs = [main_queue.popleft()]
        for index, input_spec in enumerate(node.inputs[1:], start=1):
            emission = (
                self.latest[(node.id, index)].get()
                if input_spec.semantics is InputSemantics.LATEST
                else self.flexible_current[(node.id, index)]
                if input_spec.semantics
                in {
                    InputSemantics.CONTAINS,
                    InputSemantics.OVERLAPS,
                    InputSemantics.TOLERANCE,
                }
                else self.queues[(node.id, index)].popleft()
            )
            inputs.append(emission)

        self._record_emission_gaps(node.output_port, inputs)
        if any(_has_input_overrun(item) for item in inputs) and node.gap_policy is GapPolicy.RESET:
            compiled = self.plan._compiled_kernels.get(node.id)
            if compiled is None:
                raise RuntimeError(f"MAP Node {node.id} lacks a CompiledKernel")
            self.kernel_sessions[node.id] = compiled.create_session()

        if (
            any(item.status is EmissionStatus.INVALID for item in inputs)
            and not node.accepts_invalid
        ):
            diagnostic = Diagnostic(
                Severity.WARNING,
                "INVALID_INPUT_PROPAGATED",
                "Kernel was skipped because it does not accept INVALID input",
                node_id=node.id,
                port_id=node.output_port,
                interval=main.interval,
            )
            for output_port in node.output_ports:
                if not self._output_is_demanded(output_port):
                    continue
                self._publish(
                    output_port,
                    Emission(
                        main.value,
                        main.interval,
                        self._next_sequence(output_port),
                        EmissionStatus.INVALID,
                        tuple(item for value in inputs for item in value.diagnostics)
                        + (diagnostic,),
                    ),
                )
                self._advance_frontier(output_port, main.interval.end.as_fraction())
            return True

        started_ns = time.perf_counter_ns() if self.options.profiler_enabled else None
        try:
            session = self.kernel_sessions.get(node.id)
            if session is None:
                raise RuntimeError("MAP Node lacks a CompiledKernelSession")
            result = session.run(
                tuple(item.value for item in inputs),
                RunContext(node.config, main.interval),
            )
        except Exception as error:
            raise KernelExecutionError(
                f"node {node.id} failed for interval {main.interval}: {error}"
            ) from error
        finally:
            if started_ns is not None:
                elapsed = time.perf_counter_ns() - started_ns
                self._kernel_calls[node.id] += 1
                self._kernel_total_ns[node.id] += elapsed
                self._kernel_max_ns[node.id] = max(self._kernel_max_ns[node.id], elapsed)
        inherited_status = _combined_status(inputs)
        inherited_diagnostics = tuple(item for value in inputs for item in value.diagnostics)
        if len(node.output_ports) == 1:
            if isinstance(result, KernelOutputs):
                raise KernelExecutionError(
                    f"node {node.id} port {node.output_port} returned KernelOutputs for a "
                    "single-output contract"
                )
            port_results = (result,)
        else:
            if not isinstance(result, KernelOutputs):
                raise KernelExecutionError(
                    f"node {node.id} ports {node.output_ports} requires KernelOutputs; "
                    "ordinary tuple remains a single value"
                )
            if len(result.values) != len(node.output_ports):
                raise KernelExecutionError(
                    f"node {node.id} ports {node.output_ports} returned {len(result.values)} "
                    f"outputs; contract output_count={len(node.output_ports)}"
                )
            port_results = result.values
        for output_port, port_result in zip(node.output_ports, port_results, strict=True):
            if not self._output_is_demanded(output_port):
                continue
            normalized = self._normalize_result(
                port_result,
                main.interval,
                node.id,
                output_port,
                node.max_items,
                inherited_status,
                inherited_diagnostics,
            )
            self._record_emission_gaps(output_port, normalized)
            for emission in normalized:
                self._publish(output_port, emission)
            self._advance_frontier(output_port, main.interval.end.as_fraction())
        return True

    def _output_is_demanded(self, port_id: int) -> bool:
        """Portが終端または生存consumerを持つ場合にTrueを返す。"""

        return port_id in self.root_ports or self.port_buffers[port_id].consumer_count > 0

    def _next_sequence(self, port_id: int) -> int:
        sequence = self.port_sequences[port_id]
        self.port_sequences[port_id] += 1
        return sequence

    def _normalize_result(
        self,
        result: object,
        interval: LogicalInterval,
        node_id: int,
        port_id: int,
        max_items: int,
        inherited_status: EmissionStatus,
        inherited_diagnostics: tuple[Diagnostic, ...],
    ) -> tuple[Emission[object], ...]:
        if isinstance(result, Skip):
            return ()
        values = result.values if isinstance(result, EmitMany) else (result,)
        if len(values) > max_items:
            raise KernelExecutionError(
                f"node {node_id} port {port_id} interval {interval} emitted "
                f"{len(values)} items; contract max_items={max_items}"
            )
        normalized: list[Emission[object]] = []
        for value in values:
            sequence = self._next_sequence(port_id)
            if isinstance(value, Emission):
                status = max((inherited_status, value.status), key=_status_rank)
                normalized.append(
                    Emission(
                        value.value,
                        value.interval,
                        sequence,
                        status,
                        inherited_diagnostics + value.diagnostics,
                        value.metadata,
                    )
                )
            else:
                normalized.append(
                    Emission(
                        value,
                        interval,
                        sequence,
                        inherited_status,
                        inherited_diagnostics,
                    )
                )
        return tuple(normalized)

    def _report_unmatched_inputs(self) -> None:
        for (node_id, input_index), queue in self.queues.items():
            if not queue:
                continue
            node = self.nodes_by_id[node_id]
            diagnostic = Diagnostic(
                Severity.WARNING,
                "UNMATCHED_INTERVAL_AT_EOF",
                "input emissions remained unmatched at Source EOF",
                node_id=node_id,
                port_id=node.output_port,
                interval=queue[0].interval,
                details={"input_index": input_index, "remaining_count": len(queue)},
            )
            self.diagnostics.append(diagnostic)
            self._notify_diagnostic(diagnostic)

    def _result(self) -> RunResult:
        outputs: list[OutputResult[Any]] = []
        for item in self.plan._outputs:
            snapshot = self.collectors[item.flow.port_id].snapshot()
            logical_start = snapshot.emissions[0].interval.start if snapshot.emissions else None
            logical_end = snapshot.emissions[-1].interval.end if snapshot.emissions else None
            outputs.append(
                OutputResult(
                    snapshot.emissions,
                    snapshot.info.kind,
                    snapshot.received_count,
                    snapshot.dropped_count,
                    logical_start,
                    logical_end,
                )
            )
        return RunResult(
            tuple(outputs),
            tuple(self.diagnostics),
            dict(self.status_counts),
            completed=(
                not self.stalled_nodes
                and not any(item.code == "SCHEDULER_DEADLOCK" for item in self.diagnostics)
                and not self._cancelled
                and not self.last_budget_exhausted
                and (not self.continuous or not self._active)
            ),
            profile=self._profile_snapshot() if self.options.profiler_enabled else None,
        )

    def _profile_snapshot(self) -> SessionProfile:
        """現在までのbuffer、Source、Kernel統計を不変snapshotへ変換する。"""

        descriptors = {item.buffer_id: item for item in self.plan._portable_ir.buffers}
        buffers = [
            BufferProfile(
                buffer.buffer_id,
                "port_shared",
                buffer.max_items,
                buffer.retained_count,
                buffer.high_watermark,
            )
            for buffer in self.port_buffers.values()
        ]
        buffers.extend(
            BufferProfile(
                state.history.buffer_id,
                "frame_history",
                state.history.max_items,
                len(state.history),
                state.history.high_watermark,
            )
            for state in self.frames.values()
        )
        buffers.extend(
            BufferProfile(
                ingress.buffer_id,
                "realtime_ingress",
                ingress.max_items,
                ingress.pending_count,
                ingress.high_watermark,
            )
            for ingress in self.realtime_ingresses.values()
        )
        known = {item.buffer_id for item in buffers}
        for descriptor in descriptors.values():
            if descriptor.buffer_id in known:
                continue
            buffers.append(
                BufferProfile(
                    descriptor.buffer_id,
                    descriptor.kind,
                    descriptor.max_items or 1,
                    0,
                    0,
                )
            )
        sources = []
        for node in self.plan._nodes:
            if node.kind is not NodeKind.SOURCE:
                continue
            ingress = self.realtime_ingresses.get(node.id)
            end = self.port_frontiers.get(node.output_port)
            sources.append(
                SourceProfile(
                    node.id,
                    self.published_counts[node.output_port],
                    (
                        1
                        if ingress is None and node.id in self.source_pending
                        else 0
                        if ingress is None
                        else ingress.pending_count
                    ),
                    0 if ingress is None else ingress.total_dropped_count,
                    end,
                )
            )
        return SessionProfile(
            self._scheduler_steps,
            tuple(
                KernelProfile(
                    node_id,
                    self._kernel_calls[node_id],
                    self._kernel_total_ns[node_id],
                    self._kernel_max_ns[node_id],
                )
                for node_id in sorted(self._kernel_calls)
            ),
            tuple(sorted(buffers, key=lambda item: item.buffer_id)),
            tuple(sources),
        )
