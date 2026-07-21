"""schema 0.3の限定経路を実行する最小Cython Executor。"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass

from ._cython_executor import run_f64_rate_frame
from .executor import ExecutorSession
from .graph import NodeKind, RatePolicy
from .model import Emission, EmissionStatus, LogicalInterval, LogicalTime
from .native import F64SourceValues, IdentityF64Kernel
from .runtime import ExecutionPlan, OutputResult, RunResult, RuntimeOptions


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
        source_node, rate_node, frame_node, _ = self.plan._nodes
        source = source_node.source
        if not isinstance(source, F64SourceValues):
            raise RuntimeError("validated Cython source binding was lost")
        period = rate_node.rate_period
        if period is None or frame_node.frame_size is None or frame_node.frame_hop is None:
            raise RuntimeError("validated Cython RATE/FRAME parameters were lost")
        values = array("d", source.values)
        frames, starts, ends, timebase_denominator, rate_count = run_f64_rate_frame(
            memoryview(values),
            period.numerator,
            period.denominator,
            frame_node.frame_size,
            frame_node.frame_hop,
        )
        output_spec = self.plan._outputs[0]
        collector = output_spec.collector.create_session()
        # RunResultの集計はcollector出力だけでなく、Python Executorと同じく
        # SOURCE/RATE/FRAME/MAPの各PortへpublishされたEmissionを数える。
        status_counts: Counter[EmissionStatus] = Counter()
        published_count = len(source.values) + rate_count + 2 * len(frames)
        if published_count:
            status_counts[EmissionStatus.OK] = published_count
        for sequence, (frame, start, end) in enumerate(zip(frames, starts, ends, strict=True)):
            emission = Emission(
                frame,
                LogicalInterval(
                    LogicalTime(start, 1, timebase_denominator),
                    LogicalTime(end, 1, timebase_denominator),
                ),
                sequence,
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
        if not isinstance(source_node.source, F64SourceValues):
            raise ValueError(
                "CythonExecutor contract=source_f64 requires cw.f64_source(); "
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
        if not isinstance(map_node.operation, IdentityF64Kernel):
            raise ValueError(
                "CythonExecutor contract=kernel_abi requires cw.identity_f64(); "
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
        if abi is None or not abi.native_compatible or abi.process_model != "identity_f64":
            raise ValueError(
                "CythonExecutor contract=kernel_abi requires compatible identity_f64 ABI; "
                f"node={map_node.id} port={map_node.output_port}"
            )
