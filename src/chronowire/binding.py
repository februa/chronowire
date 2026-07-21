"""PortablePlanIRをprocess-local実体へ明示的にbindする。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from .collector import Collector
from .config import Config
from .errors import ExecutionBindingError
from .extension import (
    Always,
    Every,
    EveryLogicalTime,
    Extension,
    ExtensionFailurePolicy,
    ExtensionOverflowPolicy,
    ObservationSpec,
)
from .graph import Flow, Graph, InputSemantics, InputSpec, MissingInputPolicy, NodeKind, RatePolicy
from .kernel import Backend, CompileContext, CompiledKernel, GapPolicy, Kernel
from .plan_ir import NodeDescriptor, PortablePlanIR, RationalDescriptor, TriggerDescriptor
from .runtime import ExecutionPlan, PlanSession, RunResult, RuntimeOptions, compile, output
from .source import RealtimeSource, Source


@dataclass(frozen=True)
class ExecutionBindings:
    """PortablePlanIR slotとConfig scopeへprocess-local実体を対応付ける。

    Args:
        values: binding slot IDからSource、Kernel、collector、Extensionへの完全な対応。
        configs: config scope IDから同一scope IDを持つConfigへの対応。
    """

    values: Mapping[str, object]
    configs: Mapping[str, Config] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "configs", MappingProxyType(dict(self.configs)))


@dataclass(frozen=True)
class _PrecompiledKernel:
    compiled: CompiledKernel[object]

    def compile(self, context: CompileContext) -> CompiledKernel[object]:
        del context
        return self.compiled


class BoundExecutionPlan:
    """復元ExecutionPlanと検証済みExtension bindingを保持する。"""

    def __init__(self, plan: ExecutionPlan, extensions: Mapping[str, Extension]) -> None:
        self._plan = plan
        self._extensions = MappingProxyType(dict(extensions))

    @property
    def portable_ir(self) -> PortablePlanIR:
        """bind元と同じ意味契約を持つPortablePlanIRを返す。"""

        return self._plan.portable_ir

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """検証済みbindingで一回実行する。

        Args:
            duration: 非finite pull Sourceの論理時間上限。
            options: Configと分離したruntime調整値。

        Returns:
            collector、Diagnostic、任意profileを含むRunResult。
        """

        return self._plan.create_session(extension_bindings=self._extensions).run(
            duration=duration,
            options=options,
        )

    def create_plan_session(self, *, options: RuntimeOptions | None = None) -> PlanSession:
        """検証済みbindingから継続PlanSessionを生成する。

        Args:
            options: session全体へ適用するruntime調整値。

        Returns:
            CREATED状態の新しいPlanSession。
        """

        return self._plan.create_plan_session(
            extension_bindings=self._extensions,
            options=options,
        )


def _fraction(value: RationalDescriptor) -> Fraction:
    return Fraction(value.numerator, value.denominator)


def _trigger(value: TriggerDescriptor) -> Always | Every | EveryLogicalTime:
    if value.kind == "always":
        return Always()
    if value.kind == "every" and value.count is not None:
        return Every(value.count)
    if value.kind == "every_logical_time" and value.period is not None:
        offset = RationalDescriptor(0, 1) if value.offset is None else value.offset
        return EveryLogicalTime(_fraction(value.period), _fraction(offset))
    raise ExecutionBindingError(f"unsupported trigger descriptor kind={value.kind!r}")


def _config(scope_id: str, bindings: ExecutionBindings) -> Config:
    value = bindings.configs.get(scope_id)
    if value is None:
        empty = Config()
        if empty.scope_id == scope_id:
            return empty
        raise ExecutionBindingError(
            f"missing Config binding for scope_id={scope_id!r}; contract=config_scope"
        )
    if value.scope_id != scope_id:
        raise ExecutionBindingError(
            f"Config binding scope {value.scope_id!r} does not match {scope_id!r}"
        )
    return value


def _node_parameters(
    descriptor: NodeDescriptor,
    ir: PortablePlanIR,
) -> tuple[int | None, int | None, bool, Fraction | None, RatePolicy | None]:
    frame_size, frame_hop = descriptor.frame_size, descriptor.frame_hop
    rate_period = None if descriptor.rate_period is None else _fraction(descriptor.rate_period)
    if descriptor.opcode == "frame" and (frame_size is None or frame_hop is None):
        input_port = descriptor.input_port_ids[0]
        input_time = next(item for item in ir.times if item.time_descriptor_id == input_port)
        output_time = next(
            item for item in ir.times if item.time_descriptor_id == descriptor.output_port_ids[0]
        )
        input_duration = _fraction(input_time.duration)
        input_period = _fraction(input_time.period)
        size = (_fraction(output_time.duration) - input_duration) / input_period + 1
        hop = _fraction(output_time.period) / input_period
        if size.denominator != 1 or hop.denominator != 1:
            raise ExecutionBindingError(
                f"v0.1 frame node {descriptor.node_id} parameters cannot be inferred"
            )
        frame_size, frame_hop = size.numerator, hop.numerator
    if descriptor.opcode == "rate" and rate_period is None:
        time = next(
            item for item in ir.times if item.time_descriptor_id == descriptor.output_port_ids[0]
        )
        rate_period = _fraction(time.period)
    policy = None if descriptor.rate_policy is None else RatePolicy(descriptor.rate_policy)
    if descriptor.opcode == "rate" and policy is None:
        policy = RatePolicy.HOLD
    return frame_size, frame_hop, descriptor.pad_end, rate_period, policy


def bind_plan(
    ir: PortablePlanIR,
    bindings: ExecutionBindings,
    *,
    backend: str | Backend = "python",
) -> BoundExecutionPlan:
    """PortablePlanIRを検証済みprocess-local実体へbindする。

    Args:
        ir: schema 0.1または0.2のportable plan。
        bindings: slotとConfig scopeの完全なprocess-local対応。
        backend: KernelをcompileするBackend。既定はPython。

    Returns:
        binding検証とGraph再構築を終えたBoundExecutionPlan。

    Raises:
        ExecutionBindingError: schema、slot集合、型、Config scope、Graph IDが不整合な場合。
    """

    if ir.schema_version not in {"0.1", "0.2"}:
        raise ExecutionBindingError(
            f"unsupported PortablePlanIR schema_version={ir.schema_version!r}"
        )
    required_slots = {item.slot_id for item in ir.bindings}
    missing = sorted(required_slots - set(bindings.values))
    unknown = sorted(set(bindings.values) - required_slots)
    if missing or unknown:
        raise ExecutionBindingError(
            f"binding slot mismatch; missing={missing}; unknown={unknown}; contract=exact_slots"
        )
    graph = Graph()
    flows: dict[int, Any] = {}
    edges_by_node = {
        node.node_id: sorted(
            (edge for edge in ir.edges if edge.target_node_id == node.node_id),
            key=lambda edge: edge.target_input_index,
        )
        for node in ir.nodes
    }
    for descriptor in sorted(ir.nodes, key=lambda item: item.node_id):
        config = _config(descriptor.config_scope_id, bindings)
        inputs = tuple(
            InputSpec(
                edge.source_port_id,
                InputSemantics(edge.semantics),
                edge.keyword,
                None if edge.tolerance is None else _fraction(edge.tolerance),
                MissingInputPolicy(edge.missing_policy),
            )
            for edge in edges_by_node[descriptor.node_id]
        )
        operation: Callable[..., object] | Kernel[object] | None = None
        source: Iterable[object] | Source[object] | RealtimeSource[object] | None = None
        if descriptor.opcode == "source":
            if descriptor.binding_slot is None:
                raise ExecutionBindingError(f"source node {descriptor.node_id} lacks binding slot")
            candidate = bindings.values[descriptor.binding_slot]
            if not isinstance(candidate, (Source, RealtimeSource, Iterable)):
                raise ExecutionBindingError(
                    f"slot {descriptor.binding_slot!r} node {descriptor.node_id} "
                    "requires Source or iterable"
                )
            source = candidate
        elif descriptor.opcode == "map":
            if descriptor.binding_slot is None:
                raise ExecutionBindingError(f"map node {descriptor.node_id} lacks binding slot")
            bound = bindings.values[descriptor.binding_slot]
            if isinstance(bound, CompiledKernel):
                operation = _PrecompiledKernel(bound)
            elif isinstance(bound, Kernel) or callable(bound):
                operation = bound
            else:
                raise ExecutionBindingError(
                    f"slot {descriptor.binding_slot!r} node {descriptor.node_id} requires Kernel"
                )
        frame_size, frame_hop, pad_end, rate_period, rate_policy = _node_parameters(descriptor, ir)
        ports = graph.add_node_ports(
            NodeKind(descriptor.opcode),
            config,
            inputs=inputs,
            operation=operation,
            source=source,
            frame_size=frame_size,
            frame_hop=frame_hop,
            pad_end=pad_end,
            rate_period=rate_period,
            rate_policy=rate_policy,
            max_items=descriptor.max_items,
            output_count=len(descriptor.output_port_ids),
            time_transform=descriptor.callable_time_transform,
            gap_policy=GapPolicy(descriptor.gap_policy),
        )
        if ports != descriptor.output_port_ids:
            raise ExecutionBindingError(
                f"node {descriptor.node_id} reconstructed ports {ports} do not match "
                f"{descriptor.output_port_ids}"
            )
        for port in ports:
            flows[port] = Flow._from_port(graph, port, config)
    output_specs = []
    for descriptor in ir.outputs:
        collector = bindings.values[descriptor.binding_slot]
        if not isinstance(collector, Collector):
            raise ExecutionBindingError(
                f"slot {descriptor.binding_slot!r} port {descriptor.port_id} requires Collector"
            )
        output_specs.append(output(flows[descriptor.port_id], collector=collector))
    observations = []
    extension_bindings: dict[str, Extension] = {}
    for descriptor in ir.extensions:
        binding = bindings.values[descriptor.binding_slot]
        if not isinstance(binding, Extension) or binding.abi_version != descriptor.abi_version:
            raise ExecutionBindingError(
                f"slot {descriptor.binding_slot!r} extension_id={descriptor.extension_id!r} "
                f"port={descriptor.observed_port_id} violates Extension ABI"
            )
        observations.append(
            ObservationSpec(
                flows[descriptor.observed_port_id],
                descriptor.extension_id,
                _trigger(descriptor.trigger),
                descriptor.priority,
                ExtensionFailurePolicy(descriptor.failure_policy),
                ExtensionOverflowPolicy(descriptor.overflow_policy),
                descriptor.abi_version,
            )
        )
        extension_bindings[descriptor.extension_id] = binding
    return BoundExecutionPlan(
        compile(output_specs, backend=backend, extensions=observations),
        extension_bindings,
    )
