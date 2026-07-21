"""PortablePlanIRをrun-local C++ runtimeへbindするCppExecutor。"""

from __future__ import annotations

from math import prod

from ._cpp_executor import CppNativeSession
from .collector import BufferOverflowError
from .executor import CppRuntimeMetrics, ExecutorSession
from .graph import NodeKind, RatePolicy
from .kernel import NativeRuntimeBindingProvider
from .model import Emission, EmissionStatus, LogicalInterval, LogicalTime
from .native import F64VectorSourceValues
from .runtime import ExecutionPlan, OutputResult, RunResult, RuntimeOptions

_NATIVE_TO_STATUS = (
    EmissionStatus.OK,
    EmissionStatus.DEGRADED,
    EmissionStatus.INVALID,
)
_PORTABLE_OPCODE = {"source": 0, "rate": 1, "frame": 2, "map": 3}
_COLLECTOR_KIND = {"none": 0, "latest": 1, "bounded": 2}
_OVERFLOW_POLICY = {None: 0, "fail": 0, "drop_oldest": 1, "drop_newest": 2}


def _reshape_item(
    values: memoryview[float],
    offset: int,
    shape: tuple[int, ...],
) -> object:
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
        v0.4最小実装は有限f64 vector Source、RATE、FRAME、固定CBF、一つのcollectorに限定する。
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self._last_metrics: CppRuntimeMetrics | None = None
        self._contract_context = "node=None port=None"
        self._source_emissions: tuple[Emission[tuple[float, ...]], ...]
        self._runtime = self._bind_runtime()

    @property
    def last_metrics(self) -> CppRuntimeMetrics | None:
        """直前のrunで得たC++内部計測値を返す。未実行ではNone。"""

        return self._last_metrics

    def _bind_runtime(self) -> CppNativeSession:
        nodes = self.plan._nodes
        expected = (NodeKind.SOURCE, NodeKind.RATE, NodeKind.FRAME, NodeKind.MAP)
        kinds = tuple(node.kind for node in nodes)
        if kinds != expected:
            raise ValueError(
                "CppExecutor contract=linear_native_stage requires "
                f"SOURCE->RATE->FRAME->MAP; actual={[item.value for item in kinds]}"
            )
        source_node, rate_node, frame_node, map_node = nodes
        self._contract_context = (
            f"source_node={source_node.id} source_port={source_node.output_port} "
            f"rate_node={rate_node.id} rate_port={rate_node.output_port} "
            f"frame_node={frame_node.id} frame_port={frame_node.output_port} "
            f"map_node={map_node.id} map_port={map_node.output_port}"
        )
        source = source_node.source
        if not isinstance(source, F64VectorSourceValues):
            raise ValueError(
                "CppExecutor contract=prepacked_f64_ingress requires cw.f64_vector_source(); "
                f"node={source_node.id} port={source_node.output_port}"
            )
        if rate_node.rate_policy is not RatePolicy.HOLD or rate_node.rate_period is None:
            raise ValueError(
                "CppExecutor contract=rate_policy requires HOLD with an exact period; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        if frame_node.pad_end or frame_node.frame_size is None or frame_node.frame_hop is None:
            raise ValueError(
                "CppExecutor contract=frame_eof requires fixed size/hop and pad_end=False; "
                f"node={frame_node.id} port={frame_node.output_port}"
            )
        if (
            len(self.plan._outputs) != 1
            or self.plan._outputs[0].flow.port_id != map_node.output_port
        ):
            raise ValueError(
                "CppExecutor contract=collector_boundary requires one MAP output; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if self.plan._observations:
            raise ValueError("CppExecutor contract=extension_stage is not implemented")
        ir = self.plan.portable_ir
        if ir.schema_version != "0.3":
            raise ValueError("CppExecutor contract=portable_plan_schema requires schema 0.3")
        if tuple(item.node_id for item in ir.nodes) != tuple(node.id for node in nodes) or tuple(
            item.opcode for item in ir.nodes
        ) != ("source", "rate", "frame", "map"):
            raise ValueError(
                "CppExecutor contract=portable_node_order does not match bound Plan; "
                f"{self._contract_context}"
            )
        portable_source, portable_rate, portable_frame, portable_map = ir.nodes
        if (
            portable_rate.rate_policy != "hold"
            or portable_rate.rate_period is None
            or portable_frame.frame_size is None
            or portable_frame.frame_hop is None
            or portable_frame.pad_end
        ):
            raise ValueError(
                "CppExecutor contract=portable_time_descriptor requires HOLD and fixed FRAME; "
                f"{self._contract_context}"
            )
        if any(
            port.value_schema_id == "python:opaque"
            for port in ir.ports
            if port.port_id in {node.output_port for node in nodes}
        ):
            raise ValueError(
                "CppExecutor contract=value_schema requires explicit native f64 schemas; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        compiled = self.plan._compiled_kernels.get(map_node.id)
        if not isinstance(compiled, NativeRuntimeBindingProvider):
            raise ValueError(
                "CppExecutor contract=runtime_binding requires native Kernel parameters; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        binding = compiled.create_native_runtime_binding()
        abi = next((item for item in ir.kernel_abis if item.node_id == map_node.id), None)
        if (
            abi is None
            or not abi.native_compatible
            or abi.abi_version != binding.abi_version
            or abi.process_model != binding.process_model
        ):
            raise ValueError(
                "CppExecutor contract=kernel_abi binding does not match PortablePlanIR; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if len(binding.parameter_shape) != 2:
            raise ValueError(
                "CppExecutor contract=kernel_parameter_shape requires beams x channels; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        beam_count, channel_count = binding.parameter_shape
        if channel_count != source.width:
            raise ValueError(
                "CppExecutor contract=kernel_channel_shape does not match Source width; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        self._source_emissions = source.emissions()
        if any(item.status is EmissionStatus.INVALID for item in self._source_emissions):
            raise ValueError(
                "CppExecutor contract=batch_invalid_partition is not implemented; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if any(item.metadata for item in self._source_emissions):
            raise ValueError(
                "CppExecutor contract=metadata_table is not implemented; "
                f"node={source_node.id} port={source_node.output_port}"
            )
        if any(
            diagnostic.code == "INPUT_OVERRUN"
            for emission in self._source_emissions
            for diagnostic in emission.diagnostics
        ):
            raise ValueError(
                "CppExecutor contract=gap_reset is not implemented; "
                f"node={source_node.id} port={source_node.output_port}"
            )
        output_descriptor = ir.outputs[0]
        collector_kind = _COLLECTOR_KIND.get(output_descriptor.collector_kind)
        overflow_policy = _OVERFLOW_POLICY.get(output_descriptor.overflow_policy)
        if collector_kind is None or overflow_policy is None:
            raise ValueError(
                "CppExecutor contract=native_collector supports none/latest/bounded; "
                f"port={output_descriptor.port_id}"
            )
        capacity = output_descriptor.max_items or 0
        ingress = source.native_ingress()
        try:
            return CppNativeSession(
                ir.schema_version,
                tuple(_PORTABLE_OPCODE[item.opcode] for item in ir.nodes),
                ingress.values,
                ingress.start_ticks,
                ingress.end_ticks,
                ingress.statuses,
                ingress.item_count,
                ingress.width,
                ingress.timebase_denominator,
                portable_rate.rate_period.numerator,
                portable_rate.rate_period.denominator,
                portable_frame.frame_size,
                portable_frame.frame_hop,
                binding.abi_version,
                binding.process_model,
                binding.parameter_bytes,
                beam_count,
                channel_count,
                collector_kind,
                capacity,
                overflow_policy,
                portable_source.node_id,
                portable_rate.node_id,
                portable_frame.node_id,
                portable_map.node_id,
            )
        except (ValueError, OverflowError, RuntimeError) as error:
            raise ValueError(
                f"CppExecutor failed to bind runtime: {error}; {self._contract_context}"
            ) from error

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """PortablePlanIRから構築済みのC++ state machineを一回実行する。

        Args:
            duration: 最小実装ではNoneだけを受理する。
            options: default RuntimeOptionsだけを受理する。

        Returns:
            PythonExecutorと同じ公開RunResult shape。

        Raises:
            ValueError: 未対応optionまたはnative実行契約違反の場合。
            BufferOverflowError: Bounded FAILのcapacityを超えた場合。

        境界条件:
            NoCollectではC++出力値をPythonへcopyせず、件数とstatus summaryだけを返す。
        """

        if duration is not None:
            raise ValueError("CppExecutor requires duration=None")
        if options is not None and options != RuntimeOptions():
            raise ValueError("CppExecutor does not support RuntimeOptions overrides")
        try:
            (
                output_bytes,
                output_width,
                sequences,
                starts,
                ends,
                native_statuses,
                provenance,
                timebase_denominator,
                received_count,
                dropped_count,
                overflowed,
                native_status_counts,
                metrics,
            ) = self._runtime.run()
        except (ValueError, OverflowError, RuntimeError) as error:
            raise RuntimeError(
                f"CppExecutor runtime failed: {error}; {self._contract_context}"
            ) from error
        output_descriptor = self.plan.portable_ir.outputs[0]
        if overflowed:
            raise BufferOverflowError(f"collector capacity {output_descriptor.max_items} exceeded")
        self._last_metrics = CppRuntimeMetrics(*metrics)
        map_port = next(
            port
            for port in self.plan.portable_ir.ports
            if port.port_id == output_descriptor.port_id
        )
        schema = next(
            item
            for item in self.plan.portable_ir.value_schemas
            if item.value_schema_id == map_port.value_schema_id
        )
        if schema.shape is None or prod(schema.shape) != output_width:
            raise RuntimeError(
                "CppExecutor output shape no longer matches PortablePlanIR; "
                f"{self._contract_context}"
            )
        values = memoryview(output_bytes).cast("d")
        emissions: list[Emission[object]] = []
        for retained_index, (sequence, start, end, native_status, source_indices) in enumerate(
            zip(sequences, starts, ends, native_statuses, provenance, strict=True)
        ):
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
            emissions.append(
                Emission(
                    _reshape_item(values, retained_index * output_width, schema.shape),
                    interval,
                    sequence,
                    status,
                    diagnostics,
                )
            )
        emission_tuple = tuple(emissions)
        output = OutputResult(
            emission_tuple,
            output_descriptor.collector_kind,
            received_count,
            dropped_count,
            emission_tuple[0].interval.start if emission_tuple else None,
            emission_tuple[-1].interval.end if emission_tuple else None,
        )
        status_counts = {
            status: count
            for status, count in zip(_NATIVE_TO_STATUS, native_status_counts, strict=True)
            if count
        }
        return RunResult(
            (output,),
            self.plan.diagnostics,
            status_counts,
            completed=True,
        )
