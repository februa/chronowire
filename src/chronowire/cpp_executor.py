"""PortablePlanIRをrun-local C++ runtimeへbindするCppExecutor。"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import prod
from time import perf_counter_ns
from typing import Any

from ._cpp_executor import CppCooperativeStageSession, CppGraphNativeSession
from .collector import BufferOverflowError
from .errors import ExtensionExecutionError, KernelExecutionError, PlanSessionError
from .executor import CppRuntimeMetrics, ExecutorSession
from .extension import ExtensionSession, OutputEvent, PlanContext
from .graph import NodeKind, RatePolicy
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
from .native import F64VectorSourceValues
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
    ) -> None:
        self.plan = plan
        self._extensions = extensions
        self._selected_node_ids = node_ids
        self._boundary_output_ports = boundary_output_ports
        self._last_metrics: CppRuntimeMetrics | None = None
        self._public_emission_reconstructions = 0
        self._python_boundary_dispatches = 0
        self._boundary_batch_conversions = 0
        self._contract_context = "node=None port=None"
        self._source_emissions: tuple[Emission[tuple[float, ...]], ...]
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
        if len(source_nodes) != 1:
            raise ValueError("CppExecutor contract=single_native_source")
        source_node = source_nodes[0]
        self._contract_context = (
            f"source_node={source_node.id} source_port={source_node.output_port}"
        )
        source = source_node.source
        if not isinstance(source, F64VectorSourceValues):
            raise ValueError(
                "CppExecutor contract=prepacked_f64_ingress requires cw.f64_vector_source(); "
                f"node={source_node.id} port={source_node.output_port}"
            )
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
        self._source_emissions = source.emissions()
        port_schemas = {
            port.port_id: next(
                schema
                for schema in ir.value_schemas
                if schema.value_schema_id == port.value_schema_id
            )
            for port in ir.ports
        }
        portable_nodes: list[tuple[object, ...]] = []
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
        ingress = source.native_ingress()
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
    """native prefixから末尾Python islandへ一回だけbatchを渡すmixed session。

    Args:
        plan: native prefixと末尾の単一Python StageからなるPlan。

    Raises:
        ValueError: 複数island、Pythonからnativeへの復帰、分岐または複数入力を含む場合。

    境界条件:
        v0.4の最初のmixed経路は固定shape native出力から線形なPython MAP列への
        一方向境界だけを扱う。C++ runtimeはnative prefixを自立実行し、Python関数を呼ばない。
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
        if len(python_stages) != 1 or python_stages[0] != plan.portable_ir.stages[-1]:
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
                "CppExecutor contract=mixed_stage_order requires one terminal Python island; "
                f"stage={None if stage is None else stage.stage_id} node={node_id} "
                f"port={None if node is None else node.output_port_ids[-1]} binding={binding}; "
                f"stages={tuple(item.stage_id for item in plan.portable_ir.stages)}"
            )
        stage = python_stages[0]
        nodes_by_id = {node.id: node for node in plan._nodes}
        python_nodes = tuple(nodes_by_id[node_id] for node_id in stage.node_ids)
        if not python_nodes:
            raise ValueError(f"CppExecutor contract=nonempty_python_stage; stage={stage.stage_id}")
        if len(stage.input_port_ids) != 1:
            raise ValueError(
                "CppExecutor contract=single_mixed_stage_input; "
                f"stage={stage.stage_id} node={python_nodes[0].id} "
                f"port={stage.input_port_ids} binding=None"
            )
        if stage.boundary_codec != "stream_item_v1_to_python":
            raise ValueError(
                "CppExecutor contract=mixed_stage_boundary_codec; "
                f"stage={stage.stage_id} node={python_nodes[0].id} "
                f"port={stage.input_port_ids[0]} codec={stage.boundary_codec}"
            )
        boundary_port = stage.input_port_ids[0]
        available_ports = {boundary_port}
        for node in python_nodes:
            binding = next(
                (item.slot_id for item in plan.portable_ir.bindings if item.node_id == node.id),
                None,
            )
            if (
                node.kind is not NodeKind.MAP
                or len(node.inputs) != 1
                or len(node.output_ports) != 1
                or node.inputs[0].source_port not in available_ports
            ):
                raise ValueError(
                    "CppExecutor contract=single_input_terminal_python_map_stage; "
                    f"stage={stage.stage_id} node={node.id} port={node.output_port} "
                    f"binding={binding}"
                )
            available_ports.add(node.output_port)
        python_node_ids = set(stage.node_ids)
        native_node_ids = tuple(node.id for node in plan._nodes if node.id not in python_node_ids)
        producer = next(
            (node for node in plan._nodes if boundary_port in node.output_ports),
            None,
        )
        if producer is None or producer.id not in native_node_ids:
            raise ValueError(
                "CppExecutor contract=native_to_python_boundary; "
                f"stage={stage.stage_id} node={python_nodes[0].id} port={boundary_port}"
            )
        output_ports = {item.flow.port_id for item in plan._outputs}
        if not output_ports.issubset({node.output_port for node in python_nodes}):
            raise ValueError(
                "CppExecutor contract=mixed_outputs_after_python_stage; "
                f"stage={stage.stage_id} ports={tuple(sorted(output_ports))}"
            )

        self.plan = plan
        self._stage = stage
        self._python_nodes = python_nodes
        self._boundary_port = boundary_port
        self._native = CppExecutionSession(
            plan,
            node_ids=native_node_ids,
            boundary_output_ports=(boundary_port,),
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

    @staticmethod
    def _status_max(first: EmissionStatus, second: EmissionStatus) -> EmissionStatus:
        order = {
            EmissionStatus.OK: 0,
            EmissionStatus.DEGRADED: 1,
            EmissionStatus.INVALID: 2,
        }
        return first if order[first] >= order[second] else second

    def _run_python_stage(
        self,
        boundary: tuple[Emission[object], ...],
    ) -> tuple[dict[int, tuple[Emission[object], ...]], dict[EmissionStatus, int]]:
        """境界batchを線形Python MAP列へ通し、Port別immutable batchを返す。"""

        batches: dict[int, tuple[Emission[object], ...]] = {self._boundary_port: boundary}
        counts: dict[EmissionStatus, int] = {}
        for node in self._python_nodes:
            outputs: list[Emission[object]] = []
            sequence = 0
            for item in batches[node.inputs[0].source_port]:
                if item.status is EmissionStatus.INVALID and not node.accepts_invalid:
                    diagnostic = Diagnostic(
                        Severity.WARNING,
                        "INVALID_INPUT_PROPAGATED",
                        "Kernel was skipped because it does not accept INVALID input",
                        node_id=node.id,
                        port_id=node.output_port,
                        interval=item.interval,
                    )
                    outputs.append(
                        Emission(
                            item.value,
                            item.interval,
                            sequence,
                            EmissionStatus.INVALID,
                            item.diagnostics + (diagnostic,),
                            item.metadata,
                        )
                    )
                    sequence += 1
                    continue
                try:
                    result = self._sessions[node.id].run(
                        (item.value,),
                        RunContext(node.config, item.interval),
                    )
                except Exception as error:
                    raise KernelExecutionError(
                        f"node {node.id} failed for interval {item.interval}: {error}"
                    ) from error
                if isinstance(result, KernelOutputs):
                    raise KernelExecutionError(
                        f"node {node.id} port {node.output_port} returned KernelOutputs for a "
                        "single-output contract"
                    )
                values = (
                    ()
                    if isinstance(result, Skip)
                    else (result.values if isinstance(result, EmitMany) else (result,))
                )
                if len(values) > node.max_items:
                    raise KernelExecutionError(
                        f"node {node.id} port {node.output_port} interval {item.interval} emitted "
                        f"{len(values)} items; contract max_items={node.max_items}"
                    )
                for value in values:
                    if isinstance(value, Emission):
                        output = Emission(
                            value.value,
                            value.interval,
                            sequence,
                            self._status_max(item.status, value.status),
                            item.diagnostics + value.diagnostics,
                            value.metadata,
                        )
                    else:
                        output = Emission(
                            value,
                            item.interval,
                            sequence,
                            item.status,
                            item.diagnostics,
                            item.metadata,
                        )
                    outputs.append(output)
                    sequence += 1
            batch = tuple(outputs)
            batches[node.output_port] = batch
            for item in batch:
                counts[item.status] = counts.get(item.status, 0) + 1
        return batches, counts

    def run(
        self,
        *,
        duration: float | Fraction | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """native prefixを実行し、末尾Python islandを一回のbatch dispatchで完了する。"""

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
            boundary, _, _, overflowed = self._native._decode_native_output(
                native_outputs[0],
                port_id=self._boundary_port,
            )
            if overflowed:
                raise RuntimeError(
                    "CppExecutor contract=unbounded_stage_boundary unexpectedly overflowed; "
                    f"port={self._boundary_port}"
                )
            state, stage_id = self._coordinator.advance()
            if state != 0 or stage_id != self._stage.stage_id:
                raise RuntimeError(
                    "CppExecutor contract=mixed_python_stage_advance; "
                    f"stage={self._stage.stage_id} actual_state={state} actual_stage={stage_id}"
                )
            started = perf_counter_ns()
            batches, python_counts = self._run_python_stage(boundary)
            python_stage_ns = perf_counter_ns() - started
            self._coordinator.resume(stage_id)
            completed, completed_stage = self._coordinator.advance()
            if completed != 1 or completed_stage != -1:
                raise RuntimeError(
                    "CppExecutor contract=mixed_python_stage_complete; "
                    f"stage={stage_id} actual_state={completed}"
                )
        except Exception:
            self._coordinator.abort()
            raise

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
            public_emission_reconstructions=len(boundary),
            boundary_batch_conversions=1 if boundary else 0,
            gil_acquisitions=1,
            stage_boundary_batches=1,
            stage_boundary_bytes=boundary_bytes,
            copied_batches=1 if boundary else 0,
            python_stage_ns=python_stage_ns,
            native_stage_ns=metrics[0] + metrics[1] + metrics[2],
            execution_classification="hybrid",
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
