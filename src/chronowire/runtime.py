"""Logical Graphのcompileと単一thread決定的runtimeを実装する。"""

from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
from .graph import Flow, Graph, InputSemantics, NodeKind, NodeSpec, RatePolicy
from .kernel import (
    Backend,
    CompileContext,
    CompiledKernel,
    CompiledKernelSession,
    Kernel,
    PythonBackend,
    PythonCallableKernel,
    RunContext,
)
from .model import (
    Diagnostic,
    Emission,
    EmissionStatus,
    EmitMany,
    LogicalInterval,
    LogicalTime,
    Severity,
    Skip,
)
from .plan_ir import (
    BindingDescriptor,
    BufferDescriptor,
    EdgeDescriptor,
    ExtensionDescriptor,
    NodeDescriptor,
    OutputDescriptor,
    PlanDiagnosticDescriptor,
    PortablePlanIR,
    PortDescriptor,
    RationalDescriptor,
    SourceDescriptor,
    TimeDescriptor,
    TriggerDescriptor,
)
from .runtime_buffer import CursorQueue, PortBuffer
from .source import Source, SourceRequest

T = TypeVar("T")


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


@dataclass
class _FrameState:
    items: list[Emission[object]]
    skip_remaining: int = 0


@dataclass
class _RateState:
    """一つのRATE Nodeについて次の発火時刻を保持するrun-local状態。"""

    next_fire: Fraction | None = None


def _status_rank(status: EmissionStatus) -> int:
    return {
        EmissionStatus.OK: 0,
        EmissionStatus.DEGRADED: 1,
        EmissionStatus.INVALID: 2,
    }[status]


def _combined_status(emissions: Sequence[Emission[object]]) -> EmissionStatus:
    return max((item.status for item in emissions), key=_status_rank, default=EmissionStatus.OK)


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
            signatures[node.output_port] = (Fraction(1), Fraction(1))
        elif node.kind is NodeKind.MAP:
            signatures[node.output_port] = signatures[node.inputs[0].source_port]
        elif node.kind is NodeKind.FRAME:
            input_length, input_step = signatures[node.inputs[0].source_port]
            if node.frame_size is None or node.frame_hop is None:
                raise RuntimeError("FRAME Node lacks size or hop")
            length = input_length + (node.frame_size - 1) * input_step
            signatures[node.output_port] = (length, node.frame_hop * input_step)
        elif node.kind is NodeKind.RATE:
            if node.rate_period is None:
                raise RuntimeError("RATE Node lacks period")
            signatures[node.output_port] = (node.rate_period, node.rate_period)
    return signatures


def _compile_diagnostics(nodes: Sequence[NodeSpec]) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    signatures = _time_signature(nodes)
    for node in nodes:
        for path in node.config_paths or ():
            if not node.config.has(path):
                raise MissingConfigError(f"node {node.id} requires missing Config path {path!r}")
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

    if node.kind is not NodeKind.RATE:
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

    nodes_by_port = {node.output_port: node for node in nodes}
    capacities = {node.output_port: _node_max_items(node, signatures) for node in nodes}
    reasons: dict[int, list[str]] = {
        node.output_port: [
            f"producer_burst:node={node.id}:max_items={capacities[node.output_port]}"
        ]
        for node in nodes
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
            RationalDescriptor.from_fraction(trigger.phase),
        )
    raise TypeError(
        f"extension_id {observation.extension_id!r} port {observation.flow.port_id} "
        "uses an unsupported trigger contract"
    )


def _portable_plan_ir(
    *,
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec[Any], ...],
    extensions: tuple[ObservationSpec, ...],
    diagnostics: tuple[Diagnostic, ...],
    backend_name: str,
    node_backend_names: dict[int, str],
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
            (node.output_port,),
            node.config.scope_id,
            node_backend_names[node.id],
            (
                f"source:{node.id}"
                if node.kind is NodeKind.SOURCE
                else f"kernel:{node.id}"
                if node.kind is NodeKind.MAP
                else None
            ),
            node.accepts_invalid,
            node.output_port,
            _node_max_items(node, signatures),
        )
        for node in nodes
    )
    ports = tuple(
        PortDescriptor(
            node.output_port,
            node.id,
            0,
            "python:opaque",
            node.output_port,
            f"port:{node.output_port}",
            node.output_port,
        )
        for node in nodes
    )
    buffers = tuple(
        BufferDescriptor(
            node.output_port,
            "port_shared",
            node.output_port,
            tuple(cursors_by_port[node.output_port]),
            buffer_plans[node.output_port].max_items,
            None,
            buffer_plans[node.output_port].capacity_reasons,
            buffer_plans[node.output_port].high_watermark,
            buffer_plans[node.output_port].low_watermark,
            "fail",
            "all_consumers_advanced",
            True,
        )
        for node in nodes
    )
    times = tuple(
        TimeDescriptor(
            node.output_port,
            RationalDescriptor(1, 1),
            RationalDescriptor.from_fraction(signatures[node.output_port][0]),
            RationalDescriptor.from_fraction(signatures[node.output_port][1]),
            RationalDescriptor(0, 1),
            node.kind.value,
        )
        for node in nodes
    )
    sources = tuple(
        SourceDescriptor(
            node.id,
            "pull_controlled",
            node.source.is_finite if isinstance(node.source, Source) else True,
            RationalDescriptor.from_fraction(source_request_periods[node.id]),
            None,
            None,
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
            BindingDescriptor(f"kernel:{node.id}", "kernel", node.id, node.output_port, "python-v1")
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
    return PortablePlanIR(
        schema_version="0.1",
        kind="execution_plan",
        backend=backend_name,
        nodes=plan_nodes,
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
    diagnostics = _compile_diagnostics(nodes)
    backend_instance: Backend
    if isinstance(backend, str):
        if backend != "python":
            raise ValueError(f"unsupported backend {backend!r}")
        backend_instance = PythonBackend()
    else:
        backend_instance = backend
    compiled_kernels: dict[int, CompiledKernel[object]] = {}
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
        node_backend_names[node.id] = selected_backend.name
    source_request_periods = _source_request_periods(nodes)
    sorted_extensions = tuple(sorted(extensions, key=lambda item: item.priority))
    portable_ir = _portable_plan_ir(
        nodes=nodes,
        outputs=tuple(normalized),
        extensions=sorted_extensions,
        diagnostics=diagnostics,
        backend_name=backend_instance.name,
        node_backend_names=node_backend_names,
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

    def run(self, *, duration: float | None = None) -> RunResult:
        """単一threadの決定的SchedulerでPlanを実行する。

        Args:
            duration: Sourceの論理時間上限。Noneではfinite SourceのEOFまで実行。

        Returns:
            collector結果、Diagnostic、status件数を持つRunResult。
        """

        return self.create_session().run(duration=duration)

    def create_session(
        self,
        *,
        extension_bindings: Mapping[str, Extension] | None = None,
    ) -> ExecutionSession:
        """process-local Extension実体を検証して実行instanceを生成する。

        Args:
            extension_bindings: extension_idからExtension factoryへの完全な対応。

        Returns:
            同じPlanをrun-local状態で実行するExecutionSession。

        Raises:
            ExtensionBindingError: binding不足、未知ID、種別、ABI不整合の場合。
        """

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

    def run(self, *, duration: float | None = None) -> RunResult:
        """新しいKernel、collector、Extension状態でPlanを一回実行する。

        Args:
            duration: Sourceの論理時間上限。Noneではfinite SourceのEOFまで実行。

        Returns:
            collector結果、Diagnostic、status件数を持つRunResult。
        """

        runtime = _PlanRuntime(self._plan, duration, self._extensions)
        return runtime.run()


class _PlanRuntime:
    def __init__(
        self,
        plan: ExecutionPlan,
        duration: float | None,
        extensions: tuple[_BoundExtension, ...],
    ) -> None:
        self.plan = plan
        self.duration = None if duration is None else Fraction(str(duration))
        self.nodes_by_port = {node.output_port: node for node in plan._nodes}
        self.nodes_by_id = {node.id: node for node in plan._nodes}
        signatures = _time_signature(plan._nodes)
        self.node_max_items = {node.id: _node_max_items(node, signatures) for node in plan._nodes}
        buffer_descriptors = {item.buffer_id: item for item in plan._portable_ir.buffers}
        self.port_buffers: dict[int, PortBuffer[Emission[object]]] = {
            node.output_port: PortBuffer(
                node.output_port,
                max_items=self._required_buffer_capacity(
                    node.output_port,
                    buffer_descriptors[node.output_port].max_items,
                ),
            )
            for node in plan._nodes
        }
        self.queues: dict[tuple[int, int], CursorQueue[Emission[object]]] = {}
        cursor_id = 0
        for node in plan._nodes:
            for input_index, item in enumerate(node.inputs):
                buffer = self.port_buffers[item.source_port]
                buffer.register_consumer(cursor_id)
                self.queues[(node.id, input_index)] = CursorQueue(buffer, cursor_id)
                cursor_id += 1
        self.latest: dict[tuple[int, int], Emission[object]] = {}
        self.frames: dict[int, _FrameState] = {}
        self.rates: dict[int, _RateState] = {}
        self.port_sequences: dict[int, int] = defaultdict(int)
        self.published_counts: Counter[int] = Counter()
        self.port_frontiers: dict[int, Fraction] = {}
        self.exhausted_ports: set[int] = set()
        self.stalled_nodes: set[int] = set()
        self.source_indexes: dict[int, int] = defaultdict(int)
        source_nodes = (node for node in plan._nodes if node.kind is NodeKind.SOURCE)
        self.source_iterators = {node.id: iter(self._source_values(node)) for node in source_nodes}
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

    @staticmethod
    def _required_buffer_capacity(port_id: int, max_items: int | None) -> int:
        """PortablePlanIRの通常Port capacityが実行可能な正値であることを検証する。"""

        if max_items is None or max_items <= 0:
            raise RuntimeError(
                f"port {port_id} PORT_SHARED buffer lacks a positive max_items contract"
            )
        return max_items

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
        context = PlanContext(required_node_count=len(self.plan._nodes))
        initialized: list[_ExtensionRuntime] = []
        try:
            for extension in self.extensions:
                self._initialize_extension(extension, context)
                initialized.append(extension)
            roots = tuple(
                dict.fromkeys(
                    [item.flow.port_id for item in self.plan._outputs]
                    + [extension.observation.flow.port_id for extension in self.extensions]
                )
            )
            self.root_ports = set(roots)
            seen_counts: Counter[int] = Counter()
            active = set(roots)
            while active:
                progressed = False
                for port_id in roots:
                    if port_id not in active:
                        continue
                    if seen_counts[port_id] < self.published_counts[port_id]:
                        seen_counts[port_id] += 1
                        progressed = True
                        continue
                    if port_id in self.exhausted_ports:
                        active.remove(port_id)
                        continue
                    if self._advance_port(port_id):
                        progressed = True
                    elif port_id in self.exhausted_ports:
                        active.remove(port_id)
                if active and not progressed:
                    self._report_scheduler_deadlock(active)
                    break

            self._report_unmatched_inputs()
            return self._result()
        finally:
            # collectorやKernelが失敗しても、Extensionが外部資源を閉じられるようにする。
            for extension in reversed(initialized):
                self._finalize_extension(extension, context)

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
        if not isinstance(source, Source):
            yield from source
            return
        if not source.is_finite and self.duration is None:
            raise ValueError("generated Source requires run(duration=...)")

        logical_start = LogicalTime(0)
        while True:
            if self.duration is not None and logical_start.as_fraction() >= self.duration:
                return
            request_duration = self.plan._source_request_periods.get(node.id, Fraction(1))
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
        if not self.port_buffers[port_id].can_publish(self.node_max_items[node.id]):
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

        try:
            value = next(self.source_iterators[node.id])
        except StopIteration:
            self.exhausted_ports.add(node.output_port)
            return True
        emission = self._source_emission(value, self.source_indexes[node.id])
        self.source_indexes[node.id] += 1
        if self.duration is not None and emission.interval.end.as_fraction() > self.duration:
            self.exhausted_ports.add(node.output_port)
            return True
        self._publish(node.output_port, emission)
        self._advance_frontier(node.output_port, emission.interval.end.as_fraction())
        return True

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
            if main_port in self.exhausted_ports:
                self.exhausted_ports.add(node.output_port)
                return True
            return self._advance_port(main_port)
        main = main_queue[0]

        latest_progress = self._advance_missing_latest(node, main)
        if latest_progress is not None:
            return latest_progress

        for input_index, input_spec in enumerate(node.inputs[1:], start=1):
            if input_spec.semantics is not InputSemantics.SYNCHRONOUS:
                continue
            queue = self.queues[(node.id, input_index)]
            if not queue:
                if input_spec.source_port in self.exhausted_ports:
                    self._stall_exact_merge(node, input_index, main.interval, None, None)
                    return True
                frontier = self.port_frontiers.get(input_spec.source_port)
                if frontier is not None and frontier > main.interval.start.as_fraction():
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
            self._stall_exact_merge(
                node,
                input_index,
                main.interval,
                available.interval,
                self.port_frontiers.get(input_spec.source_port),
            )
            return True
        return self._process_map_if_ready(node)

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
                self.latest[(node.id, input_index)] = queue.popleft()
                consumed = True
            if consumed:
                return True
            if queue:
                if (node.id, input_index) in self.latest:
                    continue
                self._stall_latest(node, input_index, main.interval)
                return True
            frontier = self.port_frontiers.get(input_spec.source_port)
            if frontier is not None and frontier > main.interval.start.as_fraction():
                if (node.id, input_index) in self.latest:
                    continue
                self._stall_latest(node, input_index, main.interval)
                return True
            if input_spec.source_port not in self.exhausted_ports:
                return self._advance_port(input_spec.source_port)
            if (node.id, input_index) not in self.latest:
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
        self.exhausted_ports.add(node.output_port)
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
        state = self.frames.setdefault(node.id, _FrameState([]))
        emission = queue.popleft()
        if state.skip_remaining:
            state.skip_remaining -= 1
            self._advance_frontier(node.output_port, emission.interval.end.as_fraction())
            return True
        state.items.append(emission)
        if node.frame_size is None or node.frame_hop is None:
            raise RuntimeError("FRAME Node lacks size or hop")
        if len(state.items) < node.frame_size:
            self._advance_frontier(
                node.output_port,
                state.items[0].interval.start.as_fraction(),
            )
            return True
        self._emit_frame(node, state.items[: node.frame_size])
        if node.frame_hop <= len(state.items):
            del state.items[: node.frame_hop]
        else:
            state.items.clear()
            state.skip_remaining = node.frame_hop - node.frame_size
        next_start = (
            state.items[0].interval.start.as_fraction()
            if state.items
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
        if state is None or not state.items or node.frame_size is None:
            return False
        items = list(state.items)
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
        state.items.clear()
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
        period = node.rate_period
        if period is None or node.rate_policy is not RatePolicy.HOLD:
            raise RuntimeError("RATE Node lacks a supported period and policy")
        start = source.interval.start.as_fraction()
        end = source.interval.end.as_fraction()
        state = self.rates.setdefault(node.id, _RateState())
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
                self.latest[(node.id, index)] = queue.popleft()
            if (node.id, index) not in self.latest:
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
                self.queues[(node.id, index)].popleft()
                if input_spec.semantics is InputSemantics.SYNCHRONOUS
                else self.latest[(node.id, index)]
            )
            inputs.append(emission)

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
            self._publish(
                node.output_port,
                Emission(
                    main.value,
                    main.interval,
                    self._next_sequence(node.output_port),
                    EmissionStatus.INVALID,
                    tuple(item for value in inputs for item in value.diagnostics) + (diagnostic,),
                ),
            )
            self._advance_frontier(node.output_port, main.interval.end.as_fraction())
            return True

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
        inherited_status = _combined_status(inputs)
        inherited_diagnostics = tuple(item for value in inputs for item in value.diagnostics)
        normalized = self._normalize_result(
            result,
            main.interval,
            node.id,
            node.output_port,
            node.max_items,
            inherited_status,
            inherited_diagnostics,
        )
        for emission in normalized:
            self._publish(node.output_port, emission)
        self._advance_frontier(node.output_port, main.interval.end.as_fraction())
        return True

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
            completed=True,
        )
