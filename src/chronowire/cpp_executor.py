"""PortablePlanIRをrun-local C++ runtimeへbindするCppExecutor。"""

from __future__ import annotations

from array import array
from dataclasses import replace
from fractions import Fraction
from math import isfinite, lcm, prod
from time import perf_counter_ns
from typing import Any

from ._cpp_executor import CppCooperativeStageSession, CppGraphNativeSession
from .collector import BufferOverflowError
from .errors import ExtensionExecutionError, KernelExecutionError, PlanSessionError
from .executor import CppRuntimeMetrics, ExecutorSession
from .extension import ExtensionSession, OutputEvent, PlanContext
from .graph import InputSemantics, NodeKind, RatePolicy
from .kernel import NativeRuntimeBindingProvider, RunContext
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
from .native import F64VectorSourceValues, NativeF64Ingress
from .native_module import NativeOperationRuntimeBinding
from .runtime import (
    ExecutionPlan,
    ExecutionSession,
    OutputResult,
    PlanSessionState,
    RunResult,
    RuntimeOptions,
    _BoundExtension,
)

_NATIVE_TO_STATUS = (
    EmissionStatus.OK,
    EmissionStatus.DEGRADED,
    EmissionStatus.INVALID,
)
_PORTABLE_OPCODE = {"source": 0, "rate": 1, "frame": 2, "map": 3}
_COLLECTOR_KIND = {"none": 0, "latest": 1, "bounded": 2}
_OVERFLOW_POLICY = {None: 0, "fail": 0, "drop_oldest": 1, "drop_newest": 2}
_INPUT_SEMANTICS = {"synchronous": 0, "latest": 1}
_RATE_POLICY = {None: 0, RatePolicy.HOLD: 0, RatePolicy.SAMPLE: 1}
_STATUS_TO_NATIVE = {
    EmissionStatus.OK: 0,
    EmissionStatus.DEGRADED: 1,
    EmissionStatus.INVALID: 2,
}


def _flatten_f64(value: object, shape: tuple[int, ...], *, port_id: int) -> tuple[float, ...]:
    """固定shape Python値をnative item-major f64へ検証付きで平坦化する。"""

    if not shape:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"CppExecutor contract=f64_stage_boundary; port={port_id} expected=scalar"
            )
        converted = float(value)
        if not isfinite(converted):
            raise ValueError(
                f"CppExecutor contract=f64_stage_boundary; port={port_id} value=nonfinite"
            )
        return (converted,)
    if not isinstance(value, tuple) or len(value) != shape[0]:
        raise ValueError(
            "CppExecutor contract=fixed_stage_boundary_shape; "
            f"port={port_id} expected={shape} actual_type={type(value).__name__}"
        )
    return tuple(
        item for child in value for item in _flatten_f64(child, shape[1:], port_id=port_id)
    )


def _pack_stage_ingress(
    emissions: tuple[Emission[object], ...],
    shape: tuple[int, ...],
    *,
    port_id: int,
) -> NativeF64Ingress:
    """Python Stage出力をC++所有ingressへ一回copyする。"""

    denominator = 1
    for emission in emissions:
        denominator = lcm(
            denominator,
            emission.interval.start.as_fraction().denominator,
            emission.interval.end.as_fraction().denominator,
        )
    flattened = tuple(
        value
        for emission in emissions
        for value in _flatten_f64(emission.value, shape, port_id=port_id)
    )
    values = array("d", flattened)
    starts = array(
        "q",
        (int(emission.interval.start.as_fraction() * denominator) for emission in emissions),
    )
    ends = array(
        "q",
        (int(emission.interval.end.as_fraction() * denominator) for emission in emissions),
    )
    return NativeF64Ingress(
        values.tobytes(),
        starts.tobytes(),
        ends.tobytes(),
        bytes(_STATUS_TO_NATIVE[emission.status] for emission in emissions),
        bytes(
            any(diagnostic.code == "INPUT_OVERRUN" for diagnostic in emission.diagnostics)
            for emission in emissions
        ),
        len(emissions),
        prod(shape) if shape else 1,
        denominator,
    )


def _status_max(first: EmissionStatus, second: EmissionStatus) -> EmissionStatus:
    """二つのstatusから安全側の強いstatusを返す。"""

    order = {
        EmissionStatus.OK: 0,
        EmissionStatus.DEGRADED: 1,
        EmissionStatus.INVALID: 2,
    }
    return first if order[first] >= order[second] else second


def _run_python_map_nodes(
    nodes: tuple[Any, ...],
    sessions: dict[int, Any],
    batches: dict[int, tuple[Emission[object], ...]],
) -> tuple[dict[int, tuple[Emission[object], ...]], dict[EmissionStatus, int]]:
    """topologicalなPython MAP列を名前付き入力と同じ順序でbatch実行する。

    境界条件:
        synchronous入力はprimaryと完全に同じintervalを要求し、latest入力はprimary開始時刻
        以前の最新値を選ぶ。contains/overlaps/toleranceは専用の境界codecが必要なため拒否する。
    """

    counts: dict[EmissionStatus, int] = {}
    for node in nodes:
        outputs: list[Emission[object]] = []
        sequence = 0
        input_batches = tuple(batches[item.source_port] for item in node.inputs)
        input_positions = [0] * len(input_batches)
        latest_values: dict[int, Emission[object]] = {}
        for main in input_batches[0]:
            inputs = [main]
            ready = True
            for input_index, (input_spec, input_batch) in enumerate(
                zip(node.inputs[1:], input_batches[1:], strict=True),
                start=1,
            ):
                position = input_positions[input_index]
                if input_spec.semantics is InputSemantics.SYNCHRONOUS:
                    if (
                        position >= len(input_batch)
                        or input_batch[position].interval != main.interval
                    ):
                        ready = False
                        break
                    inputs.append(input_batch[position])
                    input_positions[input_index] += 1
                    continue
                if input_spec.semantics is InputSemantics.LATEST:
                    while (
                        position < len(input_batch)
                        and input_batch[position].interval.start <= main.interval.start
                    ):
                        latest_values[input_index] = input_batch[position]
                        position += 1
                    input_positions[input_index] = position
                    latest = latest_values.get(input_index)
                    if latest is None:
                        ready = False
                        break
                    inputs.append(latest)
                    continue
                raise ValueError(
                    "CppExecutor contract=python_stage_input_semantics; "
                    f"stage=None node={node.id} port={input_spec.source_port} "
                    f"semantics={input_spec.semantics.value}"
                )
            if not ready:
                break
            inherited_status = EmissionStatus.OK
            inherited_diagnostics: tuple[Diagnostic, ...] = ()
            for item in inputs:
                inherited_status = _status_max(inherited_status, item.status)
                inherited_diagnostics += item.diagnostics
            if inherited_status is EmissionStatus.INVALID and not node.accepts_invalid:
                diagnostic = Diagnostic(
                    Severity.WARNING,
                    "INVALID_INPUT_PROPAGATED",
                    "Kernel was skipped because it does not accept INVALID input",
                    node_id=node.id,
                    port_id=node.output_port,
                    interval=main.interval,
                )
                outputs.append(
                    Emission(
                        main.value,
                        main.interval,
                        sequence,
                        EmissionStatus.INVALID,
                        inherited_diagnostics + (diagnostic,),
                        main.metadata,
                    )
                )
                sequence += 1
                continue
            try:
                result = sessions[node.id].run(
                    tuple(item.value for item in inputs),
                    RunContext(node.config, main.interval),
                )
            except Exception as error:
                raise KernelExecutionError(
                    f"node {node.id} failed for interval {main.interval}: {error}"
                ) from error
            if isinstance(result, KernelOutputs):
                raise KernelExecutionError(
                    f"node {node.id} port {node.output_port} returned KernelOutputs for a "
                    "single-output contract"
                )
            values = (
                ()
                if isinstance(result, Skip)
                else result.values
                if isinstance(result, EmitMany)
                else (result,)
            )
            if len(values) > node.max_items:
                raise KernelExecutionError(
                    f"node {node.id} port {node.output_port} interval {main.interval} emitted "
                    f"{len(values)} items; contract max_items={node.max_items}"
                )
            for value in values:
                if isinstance(value, Emission):
                    output = Emission(
                        value.value,
                        value.interval,
                        sequence,
                        _status_max(inherited_status, value.status),
                        inherited_diagnostics + value.diagnostics,
                        value.metadata,
                    )
                else:
                    output = Emission(
                        value,
                        main.interval,
                        sequence,
                        inherited_status,
                        inherited_diagnostics,
                        main.metadata,
                    )
                outputs.append(output)
                sequence += 1
        batch = tuple(outputs)
        batches[node.output_port] = batch
        for item in batch:
            counts[item.status] = counts.get(item.status, 0) + 1
    return batches, counts


def _logical_time(value: Fraction) -> LogicalTime:
    """有理数を損失なくLogicalTimeへ変換する。"""

    return LogicalTime(value.numerator, 1, value.denominator)


def _run_python_prefix_stage(
    nodes: tuple[Any, ...],
    sessions: dict[int, Any],
    source_batch: tuple[Emission[object], ...],
) -> tuple[dict[int, tuple[Emission[object], ...]], dict[EmissionStatus, int]]:
    """SOURCE/RATE/FRAME/Python MAPからなるprefix Stageをbatch実行する。"""

    source = nodes[0]
    batches: dict[int, tuple[Emission[object], ...]] = {source.output_port: source_batch}
    counts: dict[EmissionStatus, int] = {}
    for item in source_batch:
        counts[item.status] = counts.get(item.status, 0) + 1
    for node in nodes[1:]:
        input_batch = batches[node.inputs[0].source_port]
        if node.kind is NodeKind.MAP:
            batches, node_counts = _run_python_map_nodes((node,), sessions, batches)
        elif node.kind is NodeKind.RATE:
            if node.rate_period is None or node.rate_policy is None:
                raise RuntimeError(f"RATE node {node.id} lacks period/policy")
            outputs: list[Emission[object]] = []
            next_fire: Fraction | None = None
            for item in input_batch:
                start = item.interval.start.as_fraction()
                end = item.interval.end.as_fraction()
                if node.rate_policy is RatePolicy.SAMPLE:
                    if (start / node.rate_period).denominator == 1:
                        outputs.append(replace(item, sequence=len(outputs)))
                    continue
                if node.rate_policy is not RatePolicy.HOLD:
                    raise RuntimeError(f"RATE node {node.id} has unsupported policy")
                fire = start if next_fire is None else next_fire
                while fire < start:
                    skipped = -(-(start - fire) // node.rate_period)
                    fire += skipped * node.rate_period
                while fire < end:
                    outputs.append(
                        Emission(
                            item.value,
                            LogicalInterval(
                                _logical_time(fire),
                                _logical_time(fire + node.rate_period),
                            ),
                            len(outputs),
                            item.status,
                            item.diagnostics,
                            item.metadata,
                        )
                    )
                    fire += node.rate_period
                next_fire = fire
            batch = tuple(outputs)
            batches[node.output_port] = batch
            node_counts = {}
            for item in batch:
                node_counts[item.status] = node_counts.get(item.status, 0) + 1
        elif node.kind is NodeKind.FRAME:
            if node.frame_size is None or node.frame_hop is None or node.pad_end:
                raise ValueError(
                    "CppExecutor contract=python_prefix_frame requires fixed unpadded frame; "
                    f"node={node.id} port={node.output_port}"
                )
            outputs = []
            offset = 0
            while offset + node.frame_size <= len(input_batch):
                items = input_batch[offset : offset + node.frame_size]
                status = EmissionStatus.OK
                for item in items:
                    status = _status_max(status, item.status)
                outputs.append(
                    Emission(
                        tuple(item.value for item in items),
                        LogicalInterval(items[0].interval.start, items[-1].interval.end),
                        len(outputs),
                        status,
                        tuple(diagnostic for item in items for diagnostic in item.diagnostics),
                    )
                )
                offset += node.frame_hop
            batch = tuple(outputs)
            batches[node.output_port] = batch
            node_counts = {}
            for item in batch:
                node_counts[item.status] = node_counts.get(item.status, 0) + 1
        else:
            raise ValueError(
                "CppExecutor contract=python_prefix_opcode; "
                f"node={node.id} port={node.output_port} kind={node.kind.value}"
            )
        for status, count in node_counts.items():
            counts[status] = counts.get(status, 0) + count
    return batches, counts


class CppPythonStageExecutionSession(ExecutorSession):
    """all-Python Planを単一の協調的Python islandとして実行する。

    Args:
        plan: 全Stageが`python_stage` runnerを要求するExecutionPlan。
        python_session: run-local Operation/collector/Extension状態を所有するadapter側session。

    境界条件:
        v0.4最小実装はSourceからoutputまでを1つのPython islandに含む。
        C++側はstage IDのyield/resume状態だけを所有し、Python callbackを保持しない。
    """

    def __init__(self, plan: ExecutionPlan, python_session: ExecutionSession) -> None:
        stages = plan.portable_ir.stages
        if len(stages) != 1 or stages[0].execution_domain != "python":
            stage_ids = tuple(stage.stage_id for stage in stages)
            raise ValueError(
                "CppExecutor contract=single_python_island is the current cooperative scope; "
                f"stages={stage_ids}"
            )
        stage = stages[0]
        if "python_stage" not in stage.runner_capabilities:
            raise ValueError(
                "CppExecutor contract=python_stage_runner; "
                f"stage={stage.stage_id} nodes={stage.node_ids}"
            )
        self.plan = plan
        self._python_session = python_session
        self._stage_id = stage.stage_id
        self._runtime = CppCooperativeStageSession((stage.stage_id,))
        self._last_metrics: CppRuntimeMetrics | None = None

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前のPython Stage協調実行指標を返す。"""

        return self._last_metrics

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """C++ advance/yield後に最大Python islandを一回batch実行する。"""

        state, stage_id = self._runtime.advance()
        if state != 0 or stage_id != self._stage_id:
            raise RuntimeError(
                "CppExecutor contract=python_stage_advance; "
                f"stage={self._stage_id} actual_state={state} actual_stage={stage_id}"
            )
        started = perf_counter_ns()
        try:
            result = self._python_session.run(duration=duration, options=options)
        except Exception:
            self._runtime.abort()
            raise
        python_stage_ns = perf_counter_ns() - started
        self._runtime.resume(stage_id)
        completed, completed_stage = self._runtime.advance()
        if completed != 1 or completed_stage != -1:
            raise RuntimeError(
                "CppExecutor contract=python_stage_complete; "
                f"stage={stage_id} actual_state={completed}"
            )
        self._last_metrics = CppRuntimeMetrics(
            scheduler_ns=0,
            kernel_ns=0,
            output_select_ns=0,
            owned_input_bytes=0,
            output_boundary_bytes=0,
            python_native_transitions=2,
            stage_python_dispatches=1,
            executed_node_count=len(self.plan.portable_ir.nodes),
            native_run_releases_gil=False,
            gil_acquisitions=1,
            python_stage_ns=python_stage_ns,
            execution_classification="python_stage_dominated",
        )
        return result


def _reshape_item(
    values: memoryview[float],
    offset: int,
    shape: tuple[int, ...],
) -> object:
    if not shape:
        return float(values[offset])
    if len(shape) == 1:
        return tuple(float(values[offset + index]) for index in range(shape[0]))
    child_width = prod(shape[1:])
    return tuple(
        _reshape_item(values, offset + index * child_width, shape[1:]) for index in range(shape[0])
    )


class CppExecutionSession(ExecutorSession):
    """compile済みPlanを所有するrun-local C++実行instance。

    Args:
        plan: PortablePlanIR schema 0.3とprocess-local bindingを持つExecutionPlan。

    Raises:
        ValueError: CppExecutor対象外Node、ABI、Source、collectorまたは境界を含む場合。

    境界条件:
        v0.4は単一の有限f64 vector SourceからなるRATE、FRAME、native MAPのDAGを扱う。
        複数output、fan-out、version付きidentity/固定CBF ABIを実行できるが、merge、EOF padding、
        realtime push Source、任意のPython Kernelは対象外である。
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        extensions: tuple[_BoundExtension, ...] = (),
        *,
        node_ids: tuple[int, ...] | None = None,
        boundary_output_ports: tuple[int, ...] = (),
        ingress_port: int | None = None,
        ingress_emissions: tuple[Emission[object], ...] = (),
        ingress_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.plan = plan
        self._extensions = extensions
        self._selected_node_ids = node_ids
        self._boundary_output_ports = boundary_output_ports
        self._ingress_port = ingress_port
        self._ingress_emissions = ingress_emissions
        self._ingress_shape = ingress_shape
        self._last_metrics: CppRuntimeMetrics | None = None
        self._public_emission_reconstructions = 0
        self._python_boundary_dispatches = 0
        self._boundary_batch_conversions = 0
        self._contract_context = "node=None port=None"
        self._source_emissions: tuple[Emission[object], ...]
        self._node_ports: dict[int, int] = {}
        self._runtime = self._bind_runtime()

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前のrunで得たC++内部計測値を返す。未実行ではNone。"""

        return self._last_metrics

    def _bind_runtime(self) -> CppGraphNativeSession:
        selected = None if self._selected_node_ids is None else set(self._selected_node_ids)
        nodes = tuple(node for node in self.plan._nodes if selected is None or node.id in selected)
        if not nodes:
            raise ValueError("CppExecutor contract=nonempty_native_graph")
        source_nodes = tuple(node for node in nodes if node.kind is NodeKind.SOURCE)
        source: F64VectorSourceValues | None = None
        if self._ingress_port is None:
            if len(source_nodes) != 1:
                raise ValueError("CppExecutor contract=single_native_source")
            source_node = source_nodes[0]
            self._contract_context = (
                f"source_node={source_node.id} source_port={source_node.output_port}"
            )
            source_value = source_node.source
            if not isinstance(source_value, F64VectorSourceValues):
                raise ValueError(
                    "CppExecutor contract=prepacked_f64_ingress requires "
                    "cw.f64_vector_source(); "
                    f"node={source_node.id} port={source_node.output_port}"
                )
            source = source_value
        else:
            if source_nodes or self._ingress_shape is None:
                raise ValueError(
                    "CppExecutor contract=exclusive_stage_ingress; "
                    f"port={self._ingress_port} sources={tuple(node.id for node in source_nodes)}"
                )
            self._contract_context = f"stage_ingress_port={self._ingress_port}"
        ir = self.plan.portable_ir
        if ir.schema_version not in {"0.3", "0.4"}:
            raise ValueError("CppExecutor contract=portable_plan_schema requires schema 0.3/0.4")
        descriptors_by_id = {item.node_id: item for item in ir.nodes}
        if any(node.id not in descriptors_by_id for node in nodes):
            raise ValueError(
                "CppExecutor contract=portable_node_order does not match bound Plan; "
                f"{self._contract_context}"
            )
        if any(
            port.value_schema_id == "python:opaque"
            for port in ir.ports
            if port.port_id in {node.output_port for node in nodes}
        ):
            raise ValueError(
                "CppExecutor contract=value_schema requires explicit native f64 schemas; "
                f"{self._contract_context}"
            )
        port_schemas = {
            port.port_id: next(
                schema
                for schema in ir.value_schemas
                if schema.value_schema_id == port.value_schema_id
            )
            for port in ir.ports
        }
        portable_nodes: list[tuple[object, ...]] = []
        if self._ingress_port is not None and self._ingress_shape is not None:
            synthetic_node_id = max((node.id for node in self.plan._nodes), default=-1) + 1
            portable_nodes.append(
                (
                    synthetic_node_id,
                    _PORTABLE_OPCODE["source"],
                    (),
                    (),
                    self._ingress_port,
                    0,
                    1,
                    0,
                    0,
                    0,
                    False,
                    False,
                    "",
                    "",
                    b"",
                    (),
                    self._ingress_shape,
                    0,
                    0,
                    0,
                    0,
                )
            )
        for node in nodes:
            descriptor = descriptors_by_id[node.id]
            input_ports = tuple(item.source_port for item in node.inputs)
            try:
                input_semantics = tuple(
                    _INPUT_SEMANTICS[item.semantics.value] for item in node.inputs
                )
            except KeyError as error:
                raise ValueError(
                    "CppExecutor contract=native_input_semantics supports synchronous/latest; "
                    f"node={node.id} port={node.output_port}"
                ) from error
            period_numerator = 0
            period_denominator = 1
            if node.kind is NodeKind.RATE:
                if (
                    node.rate_policy not in {RatePolicy.HOLD, RatePolicy.SAMPLE}
                    or node.rate_period is None
                ):
                    raise ValueError(
                        "CppExecutor contract=rate_policy requires HOLD/SAMPLE with an exact "
                        "period; "
                        f"node={node.id} port={node.output_port}"
                    )
                period_numerator = node.rate_period.numerator
                period_denominator = node.rate_period.denominator
            if node.kind is NodeKind.FRAME and (
                node.pad_end or node.frame_size is None or node.frame_hop is None
            ):
                raise ValueError(
                    "CppExecutor contract=frame_eof requires fixed size/hop and pad_end=False; "
                    f"node={node.id} port={node.output_port}"
                )
            abi_version = ""
            process_model = ""
            parameter_bytes = b""
            parameter_shape: tuple[int, ...] = ()
            native_functions = (0, 0, 0, 0)
            if node.kind is NodeKind.MAP:
                compiled = self.plan._compiled_kernels.get(node.id)
                if not isinstance(compiled, NativeRuntimeBindingProvider):
                    stage = next(
                        (item for item in ir.stages if node.id in item.node_ids),
                        None,
                    )
                    binding = next(
                        (item for item in ir.bindings if item.node_id == node.id),
                        None,
                    )
                    raise ValueError(
                        "CppExecutor contract=runtime_binding requires native Kernel parameters; "
                        f"node={node.id} port={node.output_port} "
                        f"stage={None if stage is None else stage.stage_id} "
                        f"binding={None if binding is None else binding.slot_id}; "
                        "contract=python_stage_mixed_resume_pending"
                    )
                binding = compiled.create_native_runtime_binding()
                abi = next((item for item in ir.kernel_abis if item.node_id == node.id), None)
                if (
                    abi is None
                    or not abi.native_compatible
                    or abi.abi_version != binding.abi_version
                    or abi.process_model != binding.process_model
                ):
                    raise ValueError(
                        "CppExecutor contract=kernel_abi binding does not match PortablePlanIR; "
                        f"node={node.id} port={node.output_port}"
                    )
                abi_version = binding.abi_version
                process_model = binding.process_model
                parameter_bytes = binding.parameter_bytes
                parameter_shape = binding.parameter_shape
                if isinstance(binding, NativeOperationRuntimeBinding):
                    if binding.flush_address:
                        raise ValueError(
                            "CppExecutor contract=native_module_flush is not supported; "
                            f"node={node.id} port={node.output_port}"
                        )
                    native_functions = (
                        binding.create_address,
                        binding.process_address,
                        binding.flush_address,
                        binding.destroy_address,
                    )
            shape = port_schemas[node.output_port].shape
            if shape is None:
                raise ValueError(
                    "CppExecutor contract=fixed_output_shape; "
                    f"node={node.id} port={node.output_port}"
                )
            self._node_ports[node.id] = node.output_port
            portable_nodes.append(
                (
                    descriptor.node_id,
                    _PORTABLE_OPCODE[descriptor.opcode],
                    input_ports,
                    input_semantics,
                    node.output_port,
                    period_numerator,
                    period_denominator,
                    _RATE_POLICY[node.rate_policy],
                    node.frame_size or 0,
                    node.frame_hop or 0,
                    node.pad_end,
                    node.accepts_invalid,
                    abi_version,
                    process_model,
                    parameter_bytes,
                    parameter_shape,
                    shape,
                    *native_functions,
                )
            )
        portable_outputs: list[tuple[int, int, int, int]] = []
        if self._boundary_output_ports:
            produced_ports = {node.output_port for node in nodes}
            if self._ingress_port is not None:
                produced_ports.add(self._ingress_port)
            for port_id in self._boundary_output_ports:
                if port_id not in produced_ports:
                    raise ValueError(
                        "CppExecutor contract=native_stage_boundary_output; "
                        f"port={port_id} nodes={tuple(node.id for node in nodes)}"
                    )
                portable_outputs.append((port_id, 3, 0, 0))
        else:
            for output_descriptor in ir.outputs:
                collector_kind = _COLLECTOR_KIND.get(output_descriptor.collector_kind)
                overflow_policy = _OVERFLOW_POLICY.get(output_descriptor.overflow_policy)
                if collector_kind is None or overflow_policy is None:
                    raise ValueError(
                        "CppExecutor contract=native_collector supports none/latest/bounded; "
                        f"port={output_descriptor.port_id}"
                    )
                portable_outputs.append(
                    (
                        output_descriptor.port_id,
                        collector_kind,
                        output_descriptor.max_items or 0,
                        overflow_policy,
                    )
                )
            portable_outputs.extend(
                (observation.flow.port_id, 3, 0, 0) for observation in self.plan._observations
            )
        if self._ingress_port is None:
            if source is None:
                raise RuntimeError("CppExecutor contract=bound_native_source")
            self._source_emissions = source.emissions()
            ingress = source.native_ingress()
        else:
            if self._ingress_shape is None:
                raise RuntimeError("CppExecutor contract=bound_stage_ingress_shape")
            self._source_emissions = self._ingress_emissions
            ingress = _pack_stage_ingress(
                self._ingress_emissions,
                self._ingress_shape,
                port_id=self._ingress_port,
            )
        try:
            return CppGraphNativeSession(
                ir.schema_version,
                tuple(portable_nodes),
                tuple(portable_outputs),
                ingress.values,
                ingress.start_ticks,
                ingress.end_ticks,
                ingress.statuses,
                ingress.resets,
                ingress.item_count,
                ingress.width,
                ingress.timebase_denominator,
            )
        except (ValueError, OverflowError, RuntimeError) as error:
            raise ValueError(
                f"CppExecutor failed to bind runtime: {error}; {self._contract_context}"
            ) from error

    def _decode_native_output(
        self,
        native_output: tuple[Any, ...],
        *,
        port_id: int,
    ) -> tuple[tuple[Emission[object], ...], int, int, bool]:
        """C++ collector境界の可変shape itemを公開Emissionへ復元する。"""

        (
            output_bytes,
            value_offsets,
            shapes,
            sequences,
            starts,
            ends,
            native_statuses,
            provenance,
            invalid_nodes,
            degraded_nodes,
            native_diagnostics,
            metadata_indices,
            timebase_denominator,
            received_count,
            dropped_count,
            overflowed,
        ) = native_output
        values = memoryview(output_bytes).cast("d")
        emissions: list[Emission[object]] = []
        rows = zip(
            value_offsets[:-1],
            value_offsets[1:],
            shapes,
            sequences,
            starts,
            ends,
            native_statuses,
            provenance,
            invalid_nodes,
            degraded_nodes,
            native_diagnostics,
            metadata_indices,
            strict=True,
        )
        for (
            value_start,
            value_end,
            shape,
            sequence,
            start,
            end,
            native_status,
            source_indices,
            skipped_nodes,
            insufficient_nodes,
            module_diagnostics,
            metadata_index,
        ) in rows:
            if prod(shape) != value_end - value_start:
                raise RuntimeError(
                    f"CppExecutor output shape no longer matches its native item; port={port_id}"
                )
            status = _NATIVE_TO_STATUS[native_status]
            interval = LogicalInterval(
                LogicalTime(start, 1, timebase_denominator),
                LogicalTime(end, 1, timebase_denominator),
            )
            diagnostics = tuple(
                diagnostic
                for source_index in source_indices
                for diagnostic in self._source_emissions[source_index].diagnostics
            )
            diagnostics += tuple(
                Diagnostic(
                    Severity.WARNING,
                    "INVALID_INPUT_PROPAGATED",
                    "Kernel was skipped because it does not accept INVALID input",
                    node_id=node_id,
                    port_id=self._node_ports[node_id],
                    interval=interval,
                )
                for node_id in skipped_nodes
            )
            diagnostics += tuple(
                Diagnostic(
                    Severity.WARNING,
                    "INSUFFICIENT_INTEGRATION",
                    "MVDR covariance has fewer than channels squared samples",
                    node_id=node_id,
                    port_id=self._node_ports[node_id],
                    interval=interval,
                )
                for node_id in insufficient_nodes
            )
            severity_values = (Severity.INFO, Severity.WARNING, Severity.ERROR)
            diagnostics += tuple(
                Diagnostic(
                    severity_values[severity],
                    code,
                    message,
                    node_id=node_id,
                    port_id=self._node_ports[node_id],
                    interval=interval,
                )
                for severity, node_id, code, message in module_diagnostics
            )
            metadata = (
                self._source_emissions[metadata_index].metadata if metadata_index >= 0 else {}
            )
            emissions.append(
                Emission(
                    _reshape_item(values, value_start, shape),
                    interval,
                    sequence,
                    status,
                    diagnostics,
                    metadata,
                )
            )
        self._public_emission_reconstructions += len(emissions)
        if emissions:
            self._boundary_batch_conversions += 1
        return tuple(emissions), received_count, dropped_count, overflowed

    def _deliver_extensions(self, native_outputs: tuple[tuple[Any, ...], ...]) -> None:
        """C++観測境界のEmissionをpriority順のPython Extension Stageへ配送する。"""

        if not self._extensions:
            return
        context = PlanContext(required_node_count=len(self.plan._nodes))
        active: list[tuple[Any, ExtensionSession, Any]] = []
        first_error: Exception | None = None
        try:
            for bound in self._extensions:
                session = bound.binding.create_session()
                if not isinstance(session, ExtensionSession):
                    raise ExtensionExecutionError(
                        f"extension_id {bound.observation.extension_id!r} returned invalid "
                        "session; contract=ExtensionSession"
                    )
                trigger = bound.observation.trigger.create_session()
                try:
                    self._python_boundary_dispatches += 1
                    session.initialize(context)
                except Exception as error:
                    raise self._extension_error(bound, "initialize", error) from error
                active.append((bound, session, trigger))
            emissions_by_id = {
                observation.extension_id: self._decode_native_output(
                    native_output,
                    port_id=observation.flow.port_id,
                )[0]
                for observation, native_output in zip(
                    self.plan._observations, native_outputs, strict=True
                )
            }
            for bound, session, trigger in active:
                for emission in emissions_by_id[bound.observation.extension_id]:
                    if not trigger.should_fire(emission):
                        continue
                    try:
                        self._python_boundary_dispatches += 1
                        session.on_output(OutputEvent(bound.observation.flow.port_id, emission))
                    except Exception as error:
                        raise self._extension_error(bound, "on_output", error) from error
        except Exception as error:
            first_error = error
        finally:
            for bound, session, _ in reversed(active):
                try:
                    self._python_boundary_dispatches += 1
                    session.finalize(context)
                except Exception as error:
                    if first_error is None:
                        first_error = self._extension_error(bound, "finalize", error)
        if first_error is not None:
            raise first_error

    def _extension_error(
        self, bound: _BoundExtension, callback: str, error: Exception
    ) -> Exception:
        observation = bound.observation
        node = self.plan._graph.node_for_port(observation.flow.port_id)
        return ExtensionExecutionError(
            f"extension_id {observation.extension_id!r} "
            f"slot 'extension:{observation.extension_id}' node {node.id} "
            f"port {observation.flow.port_id} callback {callback!r} failed: {error}; "
            f"contract=failure_policy:{observation.failure_policy.value}"
        )

    def run(
        self,
        *,
        duration: float | Fraction | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """PortablePlanIRから構築済みのC++ state machineを一回実行する。

        Args:
            duration: Noneなら有限Source全体、指定時は正の排他的論理時間境界。
            options: default RuntimeOptionsだけを受理する。

        Returns:
            PythonExecutorと同じ公開RunResult shape。

        Raises:
            ValueError: 未対応optionまたはnative実行契約違反の場合。
            BufferOverflowError: Bounded FAILのcapacityを超えた場合。

        境界条件:
            NoCollectではC++出力値をPythonへcopyせず、件数とstatus summaryだけを返す。
        """

        if options is not None and options != RuntimeOptions():
            raise ValueError("CppExecutor does not support RuntimeOptions overrides")
        self._public_emission_reconstructions = 0
        self._python_boundary_dispatches = 0
        self._boundary_batch_conversions = 0
        logical_end: Fraction | None = None
        if duration is not None:
            logical_end = Fraction(str(duration))
            if logical_end <= 0:
                raise ValueError("CppExecutor duration must be positive")
        try:
            native_outputs, native_status_counts, metrics = self._runtime.run(
                None if logical_end is None else logical_end.numerator,
                None if logical_end is None else logical_end.denominator,
            )
        except (ValueError, OverflowError, RuntimeError) as error:
            raise RuntimeError(
                f"CppExecutor runtime failed: {error}; {self._contract_context}"
            ) from error
        native_metrics = CppRuntimeMetrics(
            scheduler_ns=metrics[0],
            kernel_ns=metrics[1],
            output_select_ns=metrics[2],
            owned_input_bytes=metrics[3],
            output_boundary_bytes=metrics[4],
            python_native_transitions=metrics[5],
            stage_python_dispatches=metrics[6],
            executed_node_count=metrics[7],
        )
        output_count = len(self.plan.portable_ir.outputs)
        self._deliver_extensions(native_outputs[output_count:])
        outputs: list[OutputResult[object]] = []
        for output_descriptor, native_output in zip(
            self.plan.portable_ir.outputs, native_outputs[:output_count], strict=True
        ):
            emission_tuple, received_count, dropped_count, overflowed = self._decode_native_output(
                native_output,
                port_id=output_descriptor.port_id,
            )
            if overflowed:
                raise BufferOverflowError(
                    f"collector capacity {output_descriptor.max_items} exceeded"
                )
            outputs.append(
                OutputResult(
                    emission_tuple,
                    output_descriptor.collector_kind,
                    received_count,
                    dropped_count,
                    emission_tuple[0].interval.start if emission_tuple else None,
                    emission_tuple[-1].interval.end if emission_tuple else None,
                )
            )
        status_counts = {
            status: count
            for status, count in zip(_NATIVE_TO_STATUS, native_status_counts, strict=True)
            if count
        }
        self._last_metrics = replace(
            native_metrics,
            public_emission_reconstructions=self._public_emission_reconstructions,
            python_boundary_dispatches=self._python_boundary_dispatches,
            boundary_batch_conversions=self._boundary_batch_conversions,
        )
        return RunResult(
            tuple(outputs),
            self.plan.diagnostics,
            status_counts,
            completed=True,
        )


class CppMixedExecutionSession(ExecutorSession):
    """native prefixから単一Python islandを経由してPlanを運用するmixed session。

    Args:
        plan: native prefixと単一Python Stage、任意のnative suffixからなるPlan。

    Raises:
        ValueError: 複数island、複数Python出力境界または未対応codecを含む場合。

    境界条件:
        Stage境界は固定shape f64を一回copyする。各C++ runtimeはnative区間を自立実行し、
        Python関数を直接呼ばない。
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        if plan._observations:
            raise ValueError(
                "CppExecutor contract=mixed_extension_boundary_pending; "
                "stage=None node=None port=None binding=None"
            )
        python_stages = tuple(
            stage for stage in plan.portable_ir.stages if stage.execution_domain == "python"
        )
        if len(python_stages) != 1:
            stage = python_stages[0] if python_stages else None
            node_id = None if stage is None else stage.node_ids[-1]
            node = next(
                (item for item in plan.portable_ir.nodes if item.node_id == node_id),
                None,
            )
            binding = next(
                (item.slot_id for item in plan.portable_ir.bindings if item.node_id == node_id),
                None,
            )
            raise ValueError(
                "CppExecutor contract=mixed_stage_order requires one Python island; "
                f"stage={None if stage is None else stage.stage_id} node={node_id} "
                f"port={None if node is None else node.output_port_ids[-1]} binding={binding}; "
                f"stages={tuple(item.stage_id for item in plan.portable_ir.stages)}"
            )
        stage = python_stages[0]
        nodes_by_id = {node.id: node for node in plan._nodes}
        python_nodes = tuple(nodes_by_id[node_id] for node_id in stage.node_ids)
        if not python_nodes:
            raise ValueError(f"CppExecutor contract=nonempty_python_stage; stage={stage.stage_id}")
        if not stage.input_port_ids:
            raise ValueError(
                "CppExecutor contract=nonempty_mixed_stage_input; "
                f"stage={stage.stage_id} node={python_nodes[0].id} "
                f"port={stage.input_port_ids} binding=None"
            )
        if stage.boundary_codec != "stream_item_v1_to_python":
            raise ValueError(
                "CppExecutor contract=mixed_stage_boundary_codec; "
                f"stage={stage.stage_id} node={python_nodes[0].id} "
                f"port={stage.input_port_ids[0]} codec={stage.boundary_codec}"
            )
        boundary_ports = stage.input_port_ids
        available_ports = set(boundary_ports)
        for node in python_nodes:
            binding = next(
                (item.slot_id for item in plan.portable_ir.bindings if item.node_id == node.id),
                None,
            )
            if (
                node.kind is not NodeKind.MAP
                or len(node.output_ports) != 1
                or not node.inputs
                or any(item.source_port not in available_ports for item in node.inputs)
            ):
                raise ValueError(
                    "CppExecutor contract=available_mixed_stage_inputs; "
                    f"stage={stage.stage_id} node={node.id} port={node.output_port} "
                    f"binding={binding}"
                )
            available_ports.add(node.output_port)
        python_start = next(
            index for index, node in enumerate(plan._nodes) if node.id == stage.node_ids[0]
        )
        python_end = next(
            index for index, node in enumerate(plan._nodes) if node.id == stage.node_ids[-1]
        )
        native_prefix_node_ids = tuple(node.id for node in plan._nodes[:python_start])
        native_suffix_node_ids = tuple(node.id for node in plan._nodes[python_end + 1 :])
        for boundary_port in boundary_ports:
            producer = next(
                (node for node in plan._nodes if boundary_port in node.output_ports),
                None,
            )
            if producer is None or producer.id not in native_prefix_node_ids:
                raise ValueError(
                    "CppExecutor contract=native_to_python_boundary; "
                    f"stage={stage.stage_id} node={python_nodes[0].id} port={boundary_port}"
                )
        output_ports = {item.flow.port_id for item in plan._outputs}
        if not native_suffix_node_ids and not output_ports.issubset(
            {node.output_port for node in python_nodes}
        ):
            raise ValueError(
                "CppExecutor contract=mixed_outputs_after_python_stage; "
                f"stage={stage.stage_id} ports={tuple(sorted(output_ports))}"
            )
        suffix_boundary_port: int | None = None
        suffix_boundary_shape: tuple[int, ...] | None = None
        if native_suffix_node_ids:
            if len(stage.output_port_ids) != 1:
                raise ValueError(
                    "CppExecutor contract=single_python_stage_output; "
                    f"stage={stage.stage_id} node={stage.node_ids[-1]} "
                    f"port={stage.output_port_ids} binding=None"
                )
            suffix_boundary_port = stage.output_port_ids[0]
            following = next(
                item for item in plan.portable_ir.stages if item.stage_id > stage.stage_id
            )
            if (
                suffix_boundary_port not in following.input_port_ids
                or following.boundary_codec != "python_to_stream_item_v1"
            ):
                raise ValueError(
                    "CppExecutor contract=python_to_native_stage_codec; "
                    f"stage={following.stage_id} node={following.node_ids[0]} "
                    f"port={suffix_boundary_port} codec={following.boundary_codec}"
                )
            port = next(
                item for item in plan.portable_ir.ports if item.port_id == suffix_boundary_port
            )
            schema = next(
                item
                for item in plan.portable_ir.value_schemas
                if item.value_schema_id == port.value_schema_id
            )
            if schema.dtype != "float64" or schema.shape is None:
                raise ValueError(
                    "CppExecutor contract=fixed_f64_python_to_native_boundary; "
                    f"stage={stage.stage_id} node={stage.node_ids[-1]} "
                    f"port={suffix_boundary_port} dtype={schema.dtype} shape={schema.shape}"
                )
            suffix_boundary_shape = schema.shape

        self.plan = plan
        self._stage = stage
        self._python_nodes = python_nodes
        self._boundary_ports = boundary_ports
        self._suffix_boundary_port = suffix_boundary_port
        self._suffix_boundary_shape = suffix_boundary_shape
        self._native_suffix_node_ids = native_suffix_node_ids
        self._native = CppExecutionSession(
            plan,
            node_ids=native_prefix_node_ids,
            boundary_output_ports=boundary_ports,
        )
        self._coordinator = CppCooperativeStageSession((stage.stage_id,))
        self._sessions = {
            node.id: plan._compiled_kernels[node.id].create_session() for node in python_nodes
        }
        self._last_metrics: CppRuntimeMetrics | None = None

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前のmixed Stage実行指標を返す。"""

        return self._last_metrics

    def _run_python_stage(
        self,
        boundaries: dict[int, tuple[Emission[object], ...]],
    ) -> tuple[dict[int, tuple[Emission[object], ...]], dict[EmissionStatus, int]]:
        """複数境界batchをPython MAP列へ通し、Port別immutable batchを返す。"""

        # 入力境界件数の計測対象へisland内部Portを混ぜないため、作業mappingを分離する。
        return _run_python_map_nodes(self._python_nodes, self._sessions, dict(boundaries))

    def run(
        self,
        *,
        duration: float | Fraction | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """native prefix、Python island、任意のnative suffixを順にbatch実行する。"""

        if options is not None and options != RuntimeOptions():
            raise ValueError("CppExecutor does not support RuntimeOptions overrides")
        logical_end = None if duration is None else Fraction(str(duration))
        if logical_end is not None and logical_end <= 0:
            raise ValueError("CppExecutor duration must be positive")
        try:
            native_outputs, native_counts, metrics = self._native._runtime.run(
                None if logical_end is None else logical_end.numerator,
                None if logical_end is None else logical_end.denominator,
            )
            boundaries: dict[int, tuple[Emission[object], ...]] = {}
            for boundary_port, native_output in zip(
                self._boundary_ports, native_outputs, strict=True
            ):
                boundary, _, _, overflowed = self._native._decode_native_output(
                    native_output,
                    port_id=boundary_port,
                )
                if overflowed:
                    raise RuntimeError(
                        "CppExecutor contract=unbounded_stage_boundary unexpectedly overflowed; "
                        f"port={boundary_port}"
                    )
                boundaries[boundary_port] = boundary
            state, stage_id = self._coordinator.advance()
            if state != 0 or stage_id != self._stage.stage_id:
                raise RuntimeError(
                    "CppExecutor contract=mixed_python_stage_advance; "
                    f"stage={self._stage.stage_id} actual_state={state} actual_stage={stage_id}"
                )
            started = perf_counter_ns()
            batches, python_counts = self._run_python_stage(boundaries)
            python_stage_ns = perf_counter_ns() - started
            self._coordinator.resume(stage_id)
            suffix_result: RunResult | None = None
            suffix_metrics: CppRuntimeMetrics | None = None
            suffix_boundary: tuple[Emission[object], ...] = ()
            if self._native_suffix_node_ids:
                if self._suffix_boundary_port is None or self._suffix_boundary_shape is None:
                    raise RuntimeError("CppExecutor contract=bound_native_suffix_boundary")
                suffix_boundary = batches[self._suffix_boundary_port]
                suffix = CppExecutionSession(
                    self.plan,
                    node_ids=self._native_suffix_node_ids,
                    ingress_port=self._suffix_boundary_port,
                    ingress_emissions=suffix_boundary,
                    ingress_shape=self._suffix_boundary_shape,
                )
                suffix_result = suffix.run()
                suffix_metrics = suffix.last_metrics
            completed, completed_stage = self._coordinator.advance()
            if completed != 1 or completed_stage != -1:
                raise RuntimeError(
                    "CppExecutor contract=mixed_python_stage_complete; "
                    f"stage={stage_id} actual_state={completed}"
                )
        except Exception:
            self._coordinator.abort()
            raise

        if suffix_result is not None:
            if suffix_metrics is None:
                raise RuntimeError("CppExecutor contract=native_suffix_metrics")
            suffix_boundary_counts: dict[EmissionStatus, int] = {}
            for emission in suffix_boundary:
                suffix_boundary_counts[emission.status] = (
                    suffix_boundary_counts.get(emission.status, 0) + 1
                )
            status_counts = {
                status: (
                    native_counts[index]
                    + python_counts.get(status, 0)
                    + suffix_result.status_counts.get(status, 0)
                    - suffix_boundary_counts.get(status, 0)
                )
                for index, status in enumerate(_NATIVE_TO_STATUS)
                if (
                    native_counts[index]
                    + python_counts.get(status, 0)
                    + suffix_result.status_counts.get(status, 0)
                    - suffix_boundary_counts.get(status, 0)
                )
            }
            suffix_boundary_bytes = (
                len(suffix_boundary) * prod(self._suffix_boundary_shape or ()) * 8
            )
            self._last_metrics = replace(
                suffix_metrics,
                scheduler_ns=metrics[0] + suffix_metrics.scheduler_ns,
                kernel_ns=metrics[1] + suffix_metrics.kernel_ns,
                output_select_ns=metrics[2] + suffix_metrics.output_select_ns,
                owned_input_bytes=metrics[3] + suffix_metrics.owned_input_bytes,
                python_native_transitions=suffix_metrics.python_native_transitions + 4,
                stage_python_dispatches=1,
                executed_node_count=(
                    metrics[7] + suffix_metrics.executed_node_count - 1 + len(self._python_nodes)
                ),
                gil_acquisitions=1,
                stage_boundary_batches=len(boundaries) + 1,
                stage_boundary_bytes=metrics[4] + suffix_boundary_bytes,
                copied_batches=sum(bool(batch) for batch in boundaries.values())
                + (1 if suffix_boundary else 0),
                python_stage_ns=python_stage_ns,
                native_stage_ns=(
                    metrics[0]
                    + metrics[1]
                    + metrics[2]
                    + suffix_metrics.scheduler_ns
                    + suffix_metrics.kernel_ns
                    + suffix_metrics.output_select_ns
                ),
                execution_classification="hybrid",
            )
            return replace(suffix_result, status_counts=status_counts)

        outputs: list[OutputResult[object]] = []
        for output_spec in self.plan._outputs:
            collector = output_spec.collector.create_session()
            for emission in batches[output_spec.flow.port_id]:
                collector.add(emission)
            snapshot = collector.snapshot()
            emissions = snapshot.emissions
            outputs.append(
                OutputResult(
                    emissions,
                    snapshot.info.kind,
                    snapshot.received_count,
                    snapshot.dropped_count,
                    emissions[0].interval.start if emissions else None,
                    emissions[-1].interval.end if emissions else None,
                )
            )
        status_counts = {
            status: native_counts[index] + python_counts.get(status, 0)
            for index, status in enumerate(_NATIVE_TO_STATUS)
            if native_counts[index] + python_counts.get(status, 0)
        }
        boundary_bytes = metrics[4]
        self._last_metrics = CppRuntimeMetrics(
            scheduler_ns=metrics[0],
            kernel_ns=metrics[1],
            output_select_ns=metrics[2],
            owned_input_bytes=metrics[3],
            output_boundary_bytes=boundary_bytes,
            python_native_transitions=4,
            stage_python_dispatches=1,
            executed_node_count=metrics[7] + len(self._python_nodes),
            native_run_releases_gil=True,
            public_emission_reconstructions=sum(len(batch) for batch in boundaries.values()),
            boundary_batch_conversions=sum(bool(batch) for batch in boundaries.values()),
            gil_acquisitions=1,
            stage_boundary_batches=len(boundaries),
            stage_boundary_bytes=boundary_bytes,
            copied_batches=sum(bool(batch) for batch in boundaries.values()),
            python_stage_ns=python_stage_ns,
            native_stage_ns=metrics[0] + metrics[1] + metrics[2],
            execution_classification="hybrid",
        )
        return RunResult(tuple(outputs), self.plan.diagnostics, status_counts, completed=True)


class CppPythonPrefixExecutionSession(ExecutorSession):
    """先頭Python islandのbatchをnative suffixへresumeするmixed session。

    Args:
        plan: 単一Python prefixとnative suffixからなる固定shape Plan。

    Raises:
        ValueError: Python prefixが複数Source、未対応Node、複数境界出力を含む場合。

    境界条件:
        Python Stage出力はStreamItem ABIへ検証後に一回copyする。C++ runtimeは合成SOURCEを
        ingress境界として扱うが、公開status件数ではGraph Nodeとして二重計上しない。
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        if plan._observations:
            raise ValueError(
                "CppExecutor contract=mixed_extension_boundary_pending; "
                "stage=None node=None port=None binding=None"
            )
        python_stages = tuple(
            stage for stage in plan.portable_ir.stages if stage.execution_domain == "python"
        )
        if len(python_stages) != 1 or python_stages[0] != plan.portable_ir.stages[0]:
            raise ValueError(
                "CppExecutor contract=python_prefix_stage_order; "
                f"stages={tuple(stage.stage_id for stage in plan.portable_ir.stages)}"
            )
        stage = python_stages[0]
        if len(stage.output_port_ids) != 1:
            raise ValueError(
                "CppExecutor contract=single_python_prefix_output; "
                f"stage={stage.stage_id} node={stage.node_ids[-1]} "
                f"port={stage.output_port_ids} binding=None"
            )
        following_stage = plan.portable_ir.stages[1]
        if (
            following_stage.input_port_ids != stage.output_port_ids
            or following_stage.boundary_codec != "python_to_stream_item_v1"
        ):
            raise ValueError(
                "CppExecutor contract=python_to_native_stage_codec; "
                f"stage={following_stage.stage_id} node={following_stage.node_ids[0]} "
                f"port={stage.output_port_ids[0]} codec={following_stage.boundary_codec}"
            )
        nodes_by_id = {node.id: node for node in plan._nodes}
        stage_nodes = tuple(nodes_by_id[node_id] for node_id in stage.node_ids)
        source_nodes = tuple(node for node in stage_nodes if node.kind is NodeKind.SOURCE)
        if len(source_nodes) != 1 or stage_nodes[0] is not source_nodes[0]:
            raise ValueError(
                "CppExecutor contract=single_python_prefix_source; "
                f"stage={stage.stage_id} nodes={stage.node_ids}"
            )
        source = source_nodes[0].source
        if not isinstance(source, F64VectorSourceValues):
            raise ValueError(
                "CppExecutor contract=python_prefix_f64_source; "
                f"stage={stage.stage_id} node={source_nodes[0].id} "
                f"port={source_nodes[0].output_port} binding=source:{source_nodes[0].id}"
            )
        available_ports = {source_nodes[0].output_port}
        for node in stage_nodes[1:]:
            if (
                node.kind not in {NodeKind.RATE, NodeKind.FRAME, NodeKind.MAP}
                or len(node.inputs) != 1
                or len(node.output_ports) != 1
                or node.inputs[0].source_port not in available_ports
            ):
                raise ValueError(
                    "CppExecutor contract=single_input_python_prefix_node; "
                    f"stage={stage.stage_id} node={node.id} port={node.output_port}"
                )
            available_ports.add(node.output_port)
        boundary_port = stage.output_port_ids[0]
        port = next(item for item in plan.portable_ir.ports if item.port_id == boundary_port)
        schema = next(
            item
            for item in plan.portable_ir.value_schemas
            if item.value_schema_id == port.value_schema_id
        )
        if schema.dtype != "float64" or schema.shape is None:
            raise ValueError(
                "CppExecutor contract=fixed_f64_python_to_native_boundary; "
                f"stage={stage.stage_id} node={stage.node_ids[-1]} port={boundary_port} "
                f"dtype={schema.dtype} shape={schema.shape}"
            )

        self.plan = plan
        self._stage = stage
        self._source_node = source_nodes[0]
        self._source = source
        self._stage_nodes = stage_nodes
        self._boundary_port = boundary_port
        self._boundary_shape = schema.shape
        python_node_ids = set(stage.node_ids)
        self._native_node_ids = tuple(
            node.id for node in plan._nodes if node.id not in python_node_ids
        )
        self._sessions = {
            node.id: plan._compiled_kernels[node.id].create_session()
            for node in stage_nodes
            if node.kind is NodeKind.MAP
        }
        self._coordinator = CppCooperativeStageSession((stage.stage_id,))
        self._last_metrics: CppRuntimeMetrics | None = None

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前のPython→native mixed実行指標を返す。"""

        return self._last_metrics

    def run(
        self,
        *,
        duration: float | Fraction | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """Python prefixを一回dispatchし、出力batchからnative suffixを再開する。"""

        if options is not None and options != RuntimeOptions():
            raise ValueError("CppExecutor does not support RuntimeOptions overrides")
        logical_end = None if duration is None else Fraction(str(duration))
        if logical_end is not None and logical_end <= 0:
            raise ValueError("CppExecutor duration must be positive")
        source_batch = tuple(
            emission
            for emission in self._source.emissions()
            if logical_end is None or emission.interval.end.as_fraction() <= logical_end
        )
        status_counts: dict[EmissionStatus, int] = {}
        try:
            state, stage_id = self._coordinator.advance()
            if state != 0 or stage_id != self._stage.stage_id:
                raise RuntimeError(
                    "CppExecutor contract=python_prefix_advance; "
                    f"stage={self._stage.stage_id} actual_state={state} actual_stage={stage_id}"
                )
            started = perf_counter_ns()
            batches, python_counts = _run_python_prefix_stage(
                self._stage_nodes,
                self._sessions,
                source_batch,
            )
            python_stage_ns = perf_counter_ns() - started
            boundary = batches[self._boundary_port]
            self._coordinator.resume(stage_id)
            native = CppExecutionSession(
                self.plan,
                node_ids=self._native_node_ids,
                ingress_port=self._boundary_port,
                ingress_emissions=boundary,
                ingress_shape=self._boundary_shape,
            )
            result = native.run()
            completed, completed_stage = self._coordinator.advance()
            if completed != 1 or completed_stage != -1:
                raise RuntimeError(
                    "CppExecutor contract=python_prefix_complete; "
                    f"stage={stage_id} actual_state={completed}"
                )
        except Exception:
            self._coordinator.abort()
            raise

        boundary_counts: dict[EmissionStatus, int] = {}
        for emission in boundary:
            boundary_counts[emission.status] = boundary_counts.get(emission.status, 0) + 1
        for status in _NATIVE_TO_STATUS:
            count = (
                python_counts.get(status, 0)
                + result.status_counts.get(status, 0)
                - boundary_counts.get(status, 0)
            )
            if count:
                status_counts[status] = count
            else:
                status_counts.pop(status, None)
        metrics = native.last_metrics
        if metrics is None:
            raise RuntimeError("CppExecutor contract=native_suffix_metrics")
        boundary_bytes = len(boundary) * prod(self._boundary_shape) * 8
        self._last_metrics = replace(
            metrics,
            python_native_transitions=metrics.python_native_transitions + 2,
            stage_python_dispatches=1,
            executed_node_count=metrics.executed_node_count - 1 + len(self._stage.node_ids),
            gil_acquisitions=1,
            stage_boundary_batches=1,
            stage_boundary_bytes=boundary_bytes,
            copied_batches=1 if boundary else 0,
            python_stage_ns=python_stage_ns,
            native_stage_ns=metrics.scheduler_ns + metrics.kernel_ns + metrics.output_select_ns,
            execution_classification="hybrid",
        )
        return replace(result, status_counts=status_counts)


class CppMultiIslandExecutionSession(ExecutorSession):
    """複数のPython islandをnative区間間で順次yield/resumeするsession。

    Args:
        plan: 複数入力・単一出力境界で複数Python islandを結ぶ線形Plan。

    Raises:
        ValueError: 複数境界出力、Pythonからnativeへ渡す値が非f64固定shapeの場合。

    境界条件:
        各Python islandはMAP Nodeだけを含み、islandごとにGILを一回取得する。native区間は
        C++ GraphRuntimeSessionが自立実行し、境界batchだけを一回copyする。
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        if plan._observations:
            raise ValueError(
                "CppExecutor contract=mixed_extension_boundary_pending; "
                "stage=None node=None port=None binding=None"
            )
        python_stages = tuple(
            stage for stage in plan.portable_ir.stages if stage.execution_domain == "python"
        )
        if len(python_stages) < 2:
            raise ValueError("CppExecutor contract=multiple_python_islands")
        nodes_by_id = {node.id: node for node in plan._nodes}
        node_positions = {node.id: index for index, node in enumerate(plan._nodes)}
        stage_nodes: dict[int, tuple[Any, ...]] = {}
        stage_shapes: dict[int, tuple[int, ...]] = {}
        prefix_source: F64VectorSourceValues | None = None
        for index, stage in enumerate(python_stages):
            nodes = tuple(nodes_by_id[node_id] for node_id in stage.node_ids)
            is_prefix = index == 0 and stage == plan.portable_ir.stages[0]
            if is_prefix:
                source_nodes = tuple(node for node in nodes if node.kind is NodeKind.SOURCE)
                if len(source_nodes) != 1 or nodes[0] is not source_nodes[0]:
                    raise ValueError(
                        "CppExecutor contract=multi_island_prefix_source; "
                        f"stage={stage.stage_id} nodes={stage.node_ids}"
                    )
                source = source_nodes[0].source
                if not isinstance(source, F64VectorSourceValues):
                    raise ValueError(
                        "CppExecutor contract=multi_island_prefix_f64_source; "
                        f"stage={stage.stage_id} node={source_nodes[0].id} "
                        f"port={source_nodes[0].output_port}"
                    )
                available_ports = {source_nodes[0].output_port}
                for node in nodes[1:]:
                    if (
                        node.kind not in {NodeKind.RATE, NodeKind.FRAME, NodeKind.MAP}
                        or len(node.inputs) != 1
                        or len(node.output_ports) != 1
                        or node.inputs[0].source_port not in available_ports
                    ):
                        raise ValueError(
                            "CppExecutor contract=multi_island_prefix_node; "
                            f"stage={stage.stage_id} node={node.id} port={node.output_port}"
                        )
                    available_ports.add(node.output_port)
                prefix_source = source
            elif not stage.input_port_ids or stage.boundary_codec != "stream_item_v1_to_python":
                raise ValueError(
                    "CppExecutor contract=multi_island_input_boundary; "
                    f"stage={stage.stage_id} node={stage.node_ids[0]} "
                    f"port={stage.input_port_ids} codec={stage.boundary_codec}"
                )
            available_ports = set() if is_prefix else set(stage.input_port_ids)
            for node in () if is_prefix else nodes:
                binding = next(
                    (item.slot_id for item in plan.portable_ir.bindings if item.node_id == node.id),
                    None,
                )
                if (
                    node.kind is not NodeKind.MAP
                    or len(node.output_ports) != 1
                    or not node.inputs
                    or any(item.source_port not in available_ports for item in node.inputs)
                ):
                    raise ValueError(
                        "CppExecutor contract=available_multi_island_inputs; "
                        f"stage={stage.stage_id} node={node.id} port={node.output_port} "
                        f"binding={binding}"
                    )
                available_ports.add(node.output_port)
            is_final_stage = stage == plan.portable_ir.stages[-1]
            if not is_final_stage:
                if len(stage.output_port_ids) != 1:
                    raise ValueError(
                        "CppExecutor contract=single_multi_island_output; "
                        f"stage={stage.stage_id} node={stage.node_ids[-1]} "
                        f"port={stage.output_port_ids} binding=None"
                    )
                following = plan.portable_ir.stages[stage.stage_id + 1]
                if following.boundary_codec != "python_to_stream_item_v1":
                    raise ValueError(
                        "CppExecutor contract=multi_island_output_codec; "
                        f"stage={following.stage_id} node={following.node_ids[0]} "
                        f"port={stage.output_port_ids[0]} codec={following.boundary_codec}"
                    )
                boundary_port = stage.output_port_ids[0]
                port = next(
                    item for item in plan.portable_ir.ports if item.port_id == boundary_port
                )
                schema = next(
                    item
                    for item in plan.portable_ir.value_schemas
                    if item.value_schema_id == port.value_schema_id
                )
                if schema.dtype != "float64" or schema.shape is None:
                    raise ValueError(
                        "CppExecutor contract=fixed_f64_multi_island_boundary; "
                        f"stage={stage.stage_id} node={stage.node_ids[-1]} "
                        f"port={boundary_port} dtype={schema.dtype} shape={schema.shape}"
                    )
                stage_shapes[stage.stage_id] = schema.shape
            stage_nodes[stage.stage_id] = nodes
            if index:
                previous = python_stages[index - 1]
                if node_positions[previous.node_ids[-1]] >= node_positions[stage.node_ids[0]]:
                    raise ValueError("CppExecutor contract=topological_python_islands")

        self.plan = plan
        self._python_stages = python_stages
        self._stage_nodes = stage_nodes
        self._stage_shapes = stage_shapes
        self._node_positions = node_positions
        self._prefix_source = prefix_source
        self._sessions = {
            node.id: plan._compiled_kernels[node.id].create_session()
            for stage in python_stages
            for node in stage_nodes[stage.stage_id]
            if node.kind is NodeKind.MAP
        }
        self._coordinator = CppCooperativeStageSession(
            tuple(stage.stage_id for stage in python_stages)
        )
        self._last_metrics: CppRuntimeMetrics | None = None

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前の複数island実行指標を返す。"""

        return self._last_metrics

    @staticmethod
    def _subtract_boundary_counts(
        counts: dict[EmissionStatus, int],
        boundary: tuple[Emission[object], ...],
    ) -> None:
        """合成SOURCEとして再計上された境界statusを一回分除く。"""

        for emission in boundary:
            remaining = counts.get(emission.status, 0) - 1
            if remaining:
                counts[emission.status] = remaining
            else:
                counts.pop(emission.status, None)

    @staticmethod
    def _merge_counts(
        target: dict[EmissionStatus, int],
        source: dict[EmissionStatus, int],
    ) -> None:
        for status, count in source.items():
            target[status] = target.get(status, 0) + count

    def run(
        self,
        *,
        duration: float | Fraction | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """native区間とPython islandをStage順に一回ずつ実行する。"""

        if options is not None and options != RuntimeOptions():
            raise ValueError("CppExecutor does not support RuntimeOptions overrides")
        logical_end = None if duration is None else Fraction(str(duration))
        if logical_end is not None and logical_end <= 0:
            raise ValueError("CppExecutor duration must be positive")

        status_counts: dict[EmissionStatus, int] = {}
        scheduler_ns = 0
        kernel_ns = 0
        output_select_ns = 0
        owned_input_bytes = 0
        output_boundary_bytes = 0
        executed_node_count = 0
        public_reconstructions = 0
        boundary_batches = 0
        boundary_bytes = 0
        copied_batches = 0
        python_stage_ns = 0
        native_stage_ns = 0
        cursor = 0
        ingress_port: int | None = None
        ingress_shape: tuple[int, ...] | None = None
        ingress_batch: tuple[Emission[object], ...] = ()
        final_result: RunResult | None = None
        final_batches: dict[int, tuple[Emission[object], ...]] | None = None

        try:
            for stage_index, stage in enumerate(self._python_stages):
                stage_start = self._node_positions[stage.node_ids[0]]
                native_node_ids = tuple(node.id for node in self.plan._nodes[cursor:stage_start])
                is_prefix = stage_index == 0 and stage_start == 0
                boundary: tuple[Emission[object], ...] = ()
                stage_inputs: dict[int, tuple[Emission[object], ...]] = {}
                if is_prefix:
                    if self._prefix_source is None:
                        raise RuntimeError("CppExecutor contract=bound_multi_island_prefix")
                    source_batch = tuple(
                        emission
                        for emission in self._prefix_source.emissions()
                        if logical_end is None or emission.interval.end.as_fraction() <= logical_end
                    )
                    input_port = self._stage_nodes[stage.stage_id][0].output_port
                    boundary = source_batch
                else:
                    input_ports = stage.input_port_ids
                    native = CppExecutionSession(
                        self.plan,
                        node_ids=native_node_ids,
                        boundary_output_ports=input_ports,
                        ingress_port=ingress_port,
                        ingress_emissions=ingress_batch,
                        ingress_shape=ingress_shape,
                    )
                    raw_outputs, raw_counts, raw_metrics = native._runtime.run(
                        None if cursor or logical_end is None else logical_end.numerator,
                        None if cursor or logical_end is None else logical_end.denominator,
                    )
                    native_counts = {
                        status: raw_counts[index]
                        for index, status in enumerate(_NATIVE_TO_STATUS)
                        if raw_counts[index]
                    }
                    if ingress_port is not None:
                        self._subtract_boundary_counts(native_counts, ingress_batch)
                        executed_node_count -= 1
                    self._merge_counts(status_counts, native_counts)
                    for input_port, raw_output in zip(input_ports, raw_outputs, strict=True):
                        boundary, _, _, overflowed = native._decode_native_output(
                            raw_output,
                            port_id=input_port,
                        )
                        if overflowed:
                            raise RuntimeError(
                                "CppExecutor contract=unbounded_multi_island_boundary; "
                                f"stage={stage.stage_id} port={input_port}"
                            )
                        stage_inputs[input_port] = boundary
                        public_reconstructions += len(boundary)
                        copied_batches += 1 if boundary else 0
                    scheduler_ns += raw_metrics[0]
                    kernel_ns += raw_metrics[1]
                    output_select_ns += raw_metrics[2]
                    owned_input_bytes += raw_metrics[3]
                    output_boundary_bytes += raw_metrics[4]
                    executed_node_count += raw_metrics[7]
                    native_stage_ns += raw_metrics[0] + raw_metrics[1] + raw_metrics[2]
                    boundary_batches += len(input_ports)
                    boundary_bytes += raw_metrics[4]

                state, stage_id = self._coordinator.advance()
                if state != 0 or stage_id != stage.stage_id:
                    raise RuntimeError(
                        "CppExecutor contract=multi_island_advance; "
                        f"stage={stage.stage_id} actual_state={state} actual_stage={stage_id}"
                    )
                started = perf_counter_ns()
                if is_prefix:
                    batches, python_counts = _run_python_prefix_stage(
                        self._stage_nodes[stage.stage_id],
                        self._sessions,
                        boundary,
                    )
                else:
                    batches, python_counts = _run_python_map_nodes(
                        self._stage_nodes[stage.stage_id],
                        self._sessions,
                        stage_inputs,
                    )
                python_stage_ns += perf_counter_ns() - started
                executed_node_count += len(stage.node_ids)
                self._merge_counts(status_counts, python_counts)
                self._coordinator.resume(stage_id)
                cursor = self._node_positions[stage.node_ids[-1]] + 1

                has_native_suffix = cursor < len(self.plan._nodes)
                if has_native_suffix:
                    output_port = stage.output_port_ids[0]
                    ingress_port = output_port
                    ingress_shape = self._stage_shapes[stage.stage_id]
                    ingress_batch = batches[output_port]
                    pack_bytes = len(ingress_batch) * prod(ingress_shape) * 8
                    boundary_batches += 1
                    boundary_bytes += pack_bytes
                    copied_batches += 1 if ingress_batch else 0
                else:
                    final_batches = batches

            suffix_node_ids = tuple(node.id for node in self.plan._nodes[cursor:])
            if suffix_node_ids:
                if ingress_port is None or ingress_shape is None:
                    raise RuntimeError("CppExecutor contract=bound_multi_island_suffix")
                suffix = CppExecutionSession(
                    self.plan,
                    node_ids=suffix_node_ids,
                    ingress_port=ingress_port,
                    ingress_emissions=ingress_batch,
                    ingress_shape=ingress_shape,
                )
                final_result = suffix.run()
                suffix_metrics = suffix.last_metrics
                if suffix_metrics is None:
                    raise RuntimeError("CppExecutor contract=multi_island_suffix_metrics")
                suffix_counts = dict(final_result.status_counts)
                self._subtract_boundary_counts(suffix_counts, ingress_batch)
                self._merge_counts(status_counts, suffix_counts)
                scheduler_ns += suffix_metrics.scheduler_ns
                kernel_ns += suffix_metrics.kernel_ns
                output_select_ns += suffix_metrics.output_select_ns
                owned_input_bytes += suffix_metrics.owned_input_bytes
                output_boundary_bytes += suffix_metrics.output_boundary_bytes
                executed_node_count += suffix_metrics.executed_node_count - 1
                native_stage_ns += (
                    suffix_metrics.scheduler_ns
                    + suffix_metrics.kernel_ns
                    + suffix_metrics.output_select_ns
                )
                public_reconstructions += suffix_metrics.public_emission_reconstructions

            completed, completed_stage = self._coordinator.advance()
            if completed != 1 or completed_stage != -1:
                raise RuntimeError(
                    "CppExecutor contract=multi_island_complete; "
                    f"actual_state={completed} actual_stage={completed_stage}"
                )
        except Exception:
            self._coordinator.abort()
            raise

        self._last_metrics = CppRuntimeMetrics(
            scheduler_ns=scheduler_ns,
            kernel_ns=kernel_ns,
            output_select_ns=output_select_ns,
            owned_input_bytes=owned_input_bytes,
            output_boundary_bytes=output_boundary_bytes,
            python_native_transitions=2 * (len(self._python_stages) + 1),
            stage_python_dispatches=len(self._python_stages),
            executed_node_count=executed_node_count,
            native_run_releases_gil=True,
            public_emission_reconstructions=public_reconstructions,
            boundary_batch_conversions=boundary_batches,
            gil_acquisitions=len(self._python_stages),
            stage_boundary_batches=boundary_batches,
            stage_boundary_bytes=boundary_bytes,
            copied_batches=copied_batches,
            python_stage_ns=python_stage_ns,
            native_stage_ns=native_stage_ns,
            execution_classification="hybrid",
        )
        if final_result is not None:
            return replace(final_result, status_counts=status_counts)
        if final_batches is None:
            raise RuntimeError("CppExecutor contract=multi_island_final_batches")
        outputs: list[OutputResult[object]] = []
        for output_spec in self.plan._outputs:
            collector = output_spec.collector.create_session()
            for emission in final_batches[output_spec.flow.port_id]:
                collector.add(emission)
            snapshot = collector.snapshot()
            emissions = snapshot.emissions
            outputs.append(
                OutputResult(
                    emissions,
                    snapshot.info.kind,
                    snapshot.received_count,
                    snapshot.dropped_count,
                    emissions[0].interval.start if emissions else None,
                    emissions[-1].interval.end if emissions else None,
                )
            )
        return RunResult(tuple(outputs), self.plan.diagnostics, status_counts, completed=True)


class CppPlanSession:
    """有限native Planを論理時間境界ごとにC++で継続観測するsession。

    v0.4では各境界のsnapshotを同じimmutable C++ Planから決定的に再計算する。Python側は
    lifecycleと公開RunResultだけを管理し、Scheduler、Kernel、collector選択へ介入しない。
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        options: RuntimeOptions | None,
        extensions: tuple[_BoundExtension, ...] = (),
    ) -> None:
        if extensions:
            raise PlanSessionError(
                "CppExecutor continuous Extension boundary is not supported; "
                "contract=cpp_continuous_extension"
            )
        if options is not None and options != RuntimeOptions():
            raise PlanSessionError(
                "CppExecutor PlanSession supports default RuntimeOptions only; "
                "contract=cpp_runtime_options"
            )
        self._execution = CppExecutionSession(plan)
        self._plan = plan
        self._state = PlanSessionState.CREATED
        self._logical_end: Fraction | None = None
        self._last_result: RunResult | None = None
        self._session_diagnostics: list[Diagnostic] = []
        self._source_end = max(
            (emission.interval.end.as_fraction() for emission in self._execution._source_emissions),
            default=Fraction(0),
        )

    @property
    def state(self) -> PlanSessionState:
        """現在のC++ session lifecycle状態を返す。"""

        return self._state

    def start(self) -> None:
        """run-local C++ Plan resourceを開始状態へ遷移させる。"""

        if self._state is not PlanSessionState.CREATED:
            raise PlanSessionError(
                f"CppPlanSession.start requires state=created; actual={self._state.value}"
            )
        self._session_diagnostics.append(
            Diagnostic(
                Severity.INFO,
                "SESSION_STARTED",
                "CppPlanSession acquired run-local resources",
            )
        )
        self._state = PlanSessionState.RUNNING

    def run_until(self, logical_end: LogicalTime | Fraction | int | float) -> RunResult:
        """指定した排他的論理時間境界までの累積snapshotを返す。"""

        self._require_running("run_until")
        target = self._logical_time_fraction(logical_end)
        if target <= 0 or (self._logical_end is not None and target <= self._logical_end):
            raise PlanSessionError(
                "CppPlanSession.run_until requires a positive, strictly increasing logical_end"
            )
        try:
            result = self._execution.run(duration=target)
        except Exception:
            self._state = PlanSessionState.FAILED
            raise
        self._logical_end = target
        completed = target >= self._source_end
        self._last_result = self._decorate(result, completed=completed)
        return self._last_result

    def flush(self) -> RunResult:
        """有限SourceをEOFまでC++ DAGへ流して累積snapshotを返す。"""

        self._require_running("flush")
        try:
            result = self._execution.run()
        except Exception:
            self._state = PlanSessionState.FAILED
            raise
        self._last_result = self._decorate(result, completed=True)
        return self._last_result

    def close(self) -> RunResult:
        """有限SourceをdrainしてC++ sessionをCLOSEDへ遷移させる。"""

        result = self.flush()
        self._session_diagnostics.append(
            Diagnostic(
                Severity.INFO,
                "SESSION_CLOSED",
                "CppPlanSession drained run-local resources",
            )
        )
        self._state = PlanSessionState.CLOSED
        self._last_result = self._decorate(result, completed=True)
        return self._last_result

    def cancel(self) -> RunResult:
        """未処理Sourceをdrainせず破棄してCANCELLEDへ遷移させる。"""

        self._require_running("cancel")
        self._session_diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "SESSION_CANCELLED",
                "CppPlanSession was cancelled without flushing pending input",
            )
        )
        self._state = PlanSessionState.CANCELLED
        base = self._last_result or RunResult(
            tuple(
                OutputResult((), descriptor.collector_kind, 0, 0, None, None)
                for descriptor in self._plan.portable_ir.outputs
            ),
            self._plan.diagnostics,
            {},
            completed=False,
        )
        self._last_result = self._decorate(base, completed=False)
        return self._last_result

    def _decorate(self, result: RunResult, *, completed: bool) -> RunResult:
        """session-local Diagnosticとcompletion状態を累積snapshotへ付加する。"""

        compile_count = len(self._plan.diagnostics)
        diagnostics = result.diagnostics[:compile_count] + tuple(self._session_diagnostics)
        return replace(result, diagnostics=diagnostics, completed=completed)

    def _require_running(self, operation: str) -> None:
        if self._state is not PlanSessionState.RUNNING:
            raise PlanSessionError(
                f"CppPlanSession.{operation} requires state=running; actual={self._state.value}"
            )

    @staticmethod
    def _logical_time_fraction(value: LogicalTime | Fraction | int | float) -> Fraction:
        if isinstance(value, LogicalTime):
            return value.as_fraction()
        try:
            return Fraction(str(value)) if isinstance(value, float) else Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise PlanSessionError("logical_end must be a finite rational value") from error
