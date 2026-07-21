"""schema 0.3の限定経路を実行する最小Cython Executor。"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from ._cython_executor import run_f64_rate_frame, run_f64_vector_rate_frame
from .executor import ExecutorSession
from .graph import NodeKind, NodeSpec, RatePolicy
from .kernel import NativeBatchCompiledKernel, NativeBatchKernelSession
from .model import Diagnostic, Emission, EmissionStatus, LogicalInterval, LogicalTime, Severity
from .native import (
    F64SourceValues,
    F64VectorSourceValues,
    IdentityF64Kernel,
    NativeValueBatch,
)
from .runtime import ExecutionPlan, OutputResult, RunResult, RuntimeOptions

_STATUS_TO_NATIVE = {
    EmissionStatus.OK: 0,
    EmissionStatus.DEGRADED: 1,
    EmissionStatus.INVALID: 2,
}
_NATIVE_TO_STATUS = (
    EmissionStatus.OK,
    EmissionStatus.DEGRADED,
    EmissionStatus.INVALID,
)
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _shared_timebase(emissions: tuple[Emission[object], ...], period: Fraction) -> int:
    """Source intervalとRATE periodをlossless整数tickへ揃える分母を返す。"""

    denominator = period.denominator
    for emission in emissions:
        denominator = lcm(
            denominator,
            emission.interval.start.as_fraction().denominator,
            emission.interval.end.as_fraction().denominator,
        )
    return denominator


def _ticks(value: Fraction, denominator: int, *, node_id: int, port_id: int) -> int:
    """有理時刻を検証済みsigned i64 tickへ変換する。"""

    ticks = value * denominator
    if ticks.denominator != 1 or not _I64_MIN <= ticks.numerator <= _I64_MAX:
        raise ValueError(
            "CythonExecutor contract=signed_i64_ticks cannot represent logical time; "
            f"node={node_id} port={port_id} value={value}"
        )
    return ticks.numerator


def _reshape_f64_item(
    values: memoryview[float],
    offset: int,
    shape: tuple[int, ...],
) -> object:
    """collector境界で一つの固定shape f64 itemをtupleへ復元する。"""

    if not shape:
        return float(values[offset])
    if len(shape) == 1:
        return tuple(float(values[offset + index]) for index in range(shape[0]))
    child_width = prod(shape[1:])
    return tuple(
        _reshape_f64_item(values, offset + index * child_width, shape[1:])
        for index in range(shape[0])
    )


@dataclass(frozen=True)
class CythonExecutionSession(ExecutorSession):
    """限定f64 Planを一回ずつnative loopで実行するsession。"""

    plan: ExecutionPlan

    def __post_init__(self) -> None:
        self._validate_plan()

    def run(
        self,
        *,
        duration: float | None = None,
        options: RuntimeOptions | None = None,
    ) -> RunResult:
        """SOURCE→RATE→FRAME→identity_f64→collectorを実行する。

        Args:
            duration: v0.3 prototypeではNoneだけを受理する。
            options: default RuntimeOptionsだけを受理する。

        Returns:
            PythonExecutorと同じ公開RunResult shape。

        Raises:
            ValueError: prototype対象外の実行optionの場合。
        """

        if duration is not None:
            raise ValueError("CythonExecutor prototype requires duration=None")
        if options is not None and options != RuntimeOptions():
            raise ValueError("CythonExecutor prototype does not support RuntimeOptions overrides")
        source_node, rate_node, frame_node, map_node = self.plan._nodes
        source = source_node.source
        if isinstance(source, F64VectorSourceValues):
            return self._run_vector(
                source_node,
                rate_node,
                frame_node,
                map_node,
                source,
            )
        if not isinstance(source, F64SourceValues):
            raise RuntimeError("validated Cython source binding was lost")
        period = rate_node.rate_period
        if period is None or frame_node.frame_size is None or frame_node.frame_hop is None:
            raise RuntimeError("validated Cython RATE/FRAME parameters were lost")
        source_emissions = source.emissions()
        if any(
            diagnostic.code == "INPUT_OVERRUN"
            for emission in source_emissions
            for diagnostic in emission.diagnostics
        ):
            raise ValueError(
                "CythonExecutor contract=gap_reset is not implemented; "
                f"node={source_node.id} port={source_node.output_port}"
            )
        timebase_denominator = _shared_timebase(source_emissions, period)
        if timebase_denominator > _I64_MAX:
            raise ValueError(
                "CythonExecutor contract=signed_i64_ticks cannot represent timebase; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        values = array("d", (item.value for item in source_emissions))
        start_ticks = tuple(
            _ticks(
                item.interval.start.as_fraction(),
                timebase_denominator,
                node_id=source_node.id,
                port_id=source_node.output_port,
            )
            for item in source_emissions
        )
        end_ticks = tuple(
            _ticks(
                item.interval.end.as_fraction(),
                timebase_denominator,
                node_id=source_node.id,
                port_id=source_node.output_port,
            )
            for item in source_emissions
        )
        starts = array("q", start_ticks)
        ends = array("q", end_ticks)
        source_statuses = array("B", (_STATUS_TO_NATIVE[item.status] for item in source_emissions))
        source_resets = array("B", (0 for _ in source_emissions))
        period_ticks = _ticks(
            period,
            timebase_denominator,
            node_id=rate_node.id,
            port_id=rate_node.output_port,
        )
        if any(
            end > _I64_MAX - period_ticks or end - start > _I64_MAX - period_ticks + 1
            for start, end in zip(start_ticks, end_ticks, strict=True)
        ):
            raise ValueError(
                "CythonExecutor contract=signed_i64_ticks would overflow RATE interval; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        (
            frames,
            frame_starts,
            frame_ends,
            frame_statuses,
            frame_provenance,
            returned_timebase,
            rate_status_counts,
        ) = run_f64_rate_frame(
            memoryview(values),
            memoryview(starts),
            memoryview(ends),
            memoryview(source_statuses),
            memoryview(source_resets),
            period_ticks,
            timebase_denominator,
            frame_node.frame_size,
            frame_node.frame_hop,
        )
        if returned_timebase != timebase_denominator:
            raise RuntimeError("Cython native Stage changed the shared logical timebase")
        output_spec = self.plan._outputs[0]
        collector = output_spec.collector.create_session()
        status_counts: Counter[EmissionStatus] = Counter(item.status for item in source_emissions)
        for status, count in zip(_NATIVE_TO_STATUS, rate_status_counts, strict=True):
            if count:
                status_counts[status] += count
        for sequence, (frame, start, end, native_status, provenance) in enumerate(
            zip(
                frames,
                frame_starts,
                frame_ends,
                frame_statuses,
                frame_provenance,
                strict=True,
            )
        ):
            status = _NATIVE_TO_STATUS[native_status]
            diagnostics = tuple(
                diagnostic
                for source_index in provenance
                for diagnostic in source_emissions[source_index].diagnostics
            )
            interval = LogicalInterval(
                LogicalTime(start, 1, timebase_denominator),
                LogicalTime(end, 1, timebase_denominator),
            )
            status_counts[status] += 2
            if status is EmissionStatus.INVALID and not map_node.accepts_invalid:
                diagnostics += (
                    Diagnostic(
                        Severity.WARNING,
                        "INVALID_INPUT_PROPAGATED",
                        "Kernel was skipped because it does not accept INVALID input",
                        node_id=map_node.id,
                        port_id=map_node.output_port,
                        interval=interval,
                    ),
                )
            emission = Emission(
                frame,
                interval,
                sequence,
                status,
                diagnostics,
            )
            collector.add(emission)
        snapshot = collector.snapshot()
        logical_start = snapshot.emissions[0].interval.start if snapshot.emissions else None
        logical_end = snapshot.emissions[-1].interval.end if snapshot.emissions else None
        output = OutputResult(
            snapshot.emissions,
            snapshot.info.kind,
            snapshot.received_count,
            snapshot.dropped_count,
            logical_start,
            logical_end,
        )
        return RunResult(
            (output,),
            self.plan.diagnostics,
            dict(status_counts),
            completed=True,
        )

    def _run_vector(
        self,
        source_node: NodeSpec,
        rate_node: NodeSpec,
        frame_node: NodeSpec,
        map_node: NodeSpec,
        source: F64VectorSourceValues,
    ) -> RunResult:
        """固定幅Sourceからbatch native Kernelまでを一括実行する。"""

        period = rate_node.rate_period
        if period is None or frame_node.frame_size is None or frame_node.frame_hop is None:
            raise RuntimeError("validated Cython vector RATE/FRAME parameters were lost")
        source_emissions = source.emissions()
        if any(item.status is EmissionStatus.INVALID for item in source_emissions):
            raise ValueError(
                "CythonExecutor contract=batch_invalid_partition is not implemented; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if any(
            diagnostic.code == "INPUT_OVERRUN"
            for emission in source_emissions
            for diagnostic in emission.diagnostics
        ):
            raise ValueError(
                "CythonExecutor contract=gap_reset is not implemented; "
                f"node={source_node.id} port={source_node.output_port}"
            )
        timebase_denominator = _shared_timebase(source_emissions, period)
        if timebase_denominator > _I64_MAX:
            raise ValueError(
                "CythonExecutor contract=signed_i64_ticks cannot represent timebase; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        start_ticks = tuple(
            _ticks(
                item.interval.start.as_fraction(),
                timebase_denominator,
                node_id=source_node.id,
                port_id=source_node.output_port,
            )
            for item in source_emissions
        )
        end_ticks = tuple(
            _ticks(
                item.interval.end.as_fraction(),
                timebase_denominator,
                node_id=source_node.id,
                port_id=source_node.output_port,
            )
            for item in source_emissions
        )
        period_ticks = _ticks(
            period,
            timebase_denominator,
            node_id=rate_node.id,
            port_id=rate_node.output_port,
        )
        if any(
            end > _I64_MAX - period_ticks or end - start > _I64_MAX - period_ticks + 1
            for start, end in zip(start_ticks, end_ticks, strict=True)
        ):
            raise ValueError(
                "CythonExecutor contract=signed_i64_ticks would overflow RATE interval; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        values = array(
            "d",
            (value for emission in source_emissions for value in emission.value),
        )
        starts = array("q", start_ticks)
        ends = array("q", end_ticks)
        source_statuses = array("B", (_STATUS_TO_NATIVE[item.status] for item in source_emissions))
        source_resets = array("B", (0 for _ in source_emissions))
        (
            frame_values,
            frame_count,
            frame_starts,
            frame_ends,
            frame_statuses,
            frame_provenance,
            returned_timebase,
            rate_status_counts,
        ) = run_f64_vector_rate_frame(
            memoryview(values),
            source.width,
            memoryview(starts),
            memoryview(ends),
            memoryview(source_statuses),
            memoryview(source_resets),
            period_ticks,
            timebase_denominator,
            frame_node.frame_size,
            frame_node.frame_hop,
        )
        if returned_timebase != timebase_denominator:
            raise RuntimeError("Cython vector Stage changed the shared logical timebase")
        frame_batch = NativeValueBatch(
            frame_values,
            frame_count,
            (frame_node.frame_size, source.width),
        )
        compiled = self.plan._compiled_kernels.get(map_node.id)
        if not isinstance(compiled, NativeBatchCompiledKernel):
            raise RuntimeError("validated native batch CompiledKernel binding was lost")
        kernel_session = compiled.create_session()
        if not isinstance(kernel_session, NativeBatchKernelSession):
            raise RuntimeError("native batch CompiledKernel returned an incompatible session")
        kernel_result = kernel_session.run_batch(
            frame_batch.f64_view(),
            item_count=frame_batch.item_count,
            item_shape=frame_batch.item_shape,
        )
        if not isinstance(kernel_result, NativeValueBatch):
            raise RuntimeError("native batch Kernel must return NativeValueBatch")
        if kernel_result.item_count != frame_count:
            raise RuntimeError("native batch Kernel changed the logical item count")

        output_spec = self.plan._outputs[0]
        collector = output_spec.collector.create_session()
        status_counts: Counter[EmissionStatus] = Counter(item.status for item in source_emissions)
        for status, count in zip(_NATIVE_TO_STATUS, rate_status_counts, strict=True):
            if count:
                status_counts[status] += count
        output_values = kernel_result.f64_view()
        output_width = prod(kernel_result.item_shape)
        for sequence, (start, end, native_status, provenance) in enumerate(
            zip(
                frame_starts,
                frame_ends,
                frame_statuses,
                frame_provenance,
                strict=True,
            )
        ):
            status = _NATIVE_TO_STATUS[native_status]
            diagnostics = tuple(
                diagnostic
                for source_index in provenance
                for diagnostic in source_emissions[source_index].diagnostics
            )
            interval = LogicalInterval(
                LogicalTime(start, 1, timebase_denominator),
                LogicalTime(end, 1, timebase_denominator),
            )
            status_counts[status] += 2
            collector.add(
                Emission(
                    _reshape_f64_item(
                        output_values,
                        sequence * output_width,
                        kernel_result.item_shape,
                    ),
                    interval,
                    sequence,
                    status,
                    diagnostics,
                )
            )
        snapshot = collector.snapshot()
        logical_start = snapshot.emissions[0].interval.start if snapshot.emissions else None
        logical_end = snapshot.emissions[-1].interval.end if snapshot.emissions else None
        output = OutputResult(
            snapshot.emissions,
            snapshot.info.kind,
            snapshot.received_count,
            snapshot.dropped_count,
            logical_start,
            logical_end,
        )
        return RunResult(
            (output,),
            self.plan.diagnostics,
            dict(status_counts),
            completed=True,
        )

    def _validate_plan(self) -> None:
        """推測なしで実行できる最小schema 0.3経路だけを受理する。"""

        nodes = self.plan._nodes
        kinds = tuple(node.kind for node in nodes)
        expected = (NodeKind.SOURCE, NodeKind.RATE, NodeKind.FRAME, NodeKind.MAP)
        if kinds != expected:
            raise ValueError(
                "CythonExecutor contract=linear_native_stage requires "
                "SOURCE->RATE->FRAME->MAP; "
                f"actual={[item.value for item in kinds]}"
            )
        source_node, rate_node, frame_node, map_node = nodes
        is_scalar_source = isinstance(source_node.source, F64SourceValues)
        is_vector_source = isinstance(source_node.source, F64VectorSourceValues)
        if not is_scalar_source and not is_vector_source:
            raise ValueError(
                "CythonExecutor contract=source_f64 requires cw.f64_source() or "
                "cw.f64_vector_source(); "
                f"node={source_node.id} port={source_node.output_port}"
            )
        if rate_node.rate_policy is not RatePolicy.HOLD:
            raise ValueError(
                "CythonExecutor contract=rate_policy requires HOLD; "
                f"node={rate_node.id} port={rate_node.output_port}"
            )
        if frame_node.pad_end:
            raise ValueError(
                "CythonExecutor contract=frame_eof forbids pad_end; "
                f"node={frame_node.id} port={frame_node.output_port}"
            )
        compiled = self.plan._compiled_kernels.get(map_node.id)
        if is_scalar_source and not isinstance(map_node.operation, IdentityF64Kernel):
            raise ValueError(
                "CythonExecutor contract=kernel_abi requires cw.identity_f64(); "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if is_vector_source and not isinstance(compiled, NativeBatchCompiledKernel):
            raise ValueError(
                "CythonExecutor contract=batch_kernel_abi requires a native batch Kernel; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if (
            len(self.plan._outputs) != 1
            or self.plan._outputs[0].flow.port_id != map_node.output_port
        ):
            raise ValueError(
                "CythonExecutor contract=collector_boundary requires one MAP output; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if self.plan._observations:
            raise ValueError("CythonExecutor prototype does not support Extension boundaries")
        ir = self.plan.portable_ir
        if ir.schema_version != "0.3":
            raise ValueError("CythonExecutor prototype requires PortablePlanIR schema 0.3")
        if any(
            port.value_schema_id == "python:opaque"
            for port in ir.ports
            if port.port_id in {node.output_port for node in nodes}
        ):
            raise ValueError(
                "CythonExecutor contract=value_schema requires explicit native f64 schemas; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        abi = next((item for item in ir.kernel_abis if item.node_id == map_node.id), None)
        if abi is None or not abi.native_compatible:
            raise ValueError(
                "CythonExecutor contract=kernel_abi requires a native-compatible ABI; "
                f"node={map_node.id} port={map_node.output_port}"
            )
        if is_scalar_source and abi.process_model != "identity_f64":
            raise ValueError(
                "CythonExecutor contract=kernel_abi requires compatible identity_f64 ABI; "
                f"node={map_node.id} port={map_node.output_port}"
            )
