"""Logical Graphのcompileと単一thread決定的runtimeを実装する。"""

from __future__ import annotations

import inspect
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, TypeVar

from .collector import Collector, CollectorSession, NoCollect
from .errors import DuplicateOutputError, KernelExecutionError, MissingConfigError
from .extension import Extension, OutputEvent, PlanContext
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


def compile(
    outputs: Sequence[Flow[Any] | OutputSpec[Any]],
    *,
    backend: str | Backend = "python",
    extensions: Sequence[Extension] = (),
) -> ExecutionPlan:
    """Flow群から不変なExecutionPlanを生成する。

    Args:
        outputs: 観測終端。bare FlowはNoCollectとして実行だけ行う。
        extensions: collectorと独立してPortを観測するExtension。

    Raises:
        ValueError: outputsが空、または異なるGraphを含む場合。
        DuplicateOutputError: 同じPortが複数回指定された場合。
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

    observed_ports = [port for extension in extensions for port in extension.observed_ports()]
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
    return ExecutionPlan(
        graph=graph,
        nodes=nodes,
        outputs=tuple(normalized),
        extensions=tuple(sorted(extensions, key=lambda item: item.priority)),
        compile_diagnostics=diagnostics,
        compiled_kernels=compiled_kernels,
        backend_name=backend_instance.name,
        node_backend_names=node_backend_names,
        source_request_periods=_source_request_periods(nodes),
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
        extensions: tuple[Extension, ...],
        compile_diagnostics: tuple[Diagnostic, ...],
        compiled_kernels: dict[int, CompiledKernel[object]],
        backend_name: str,
        node_backend_names: dict[int, str],
        source_request_periods: dict[int, Fraction],
    ) -> None:
        self._graph = graph
        self._nodes = nodes
        self._outputs = outputs
        self._extensions = extensions
        self._compile_diagnostics = compile_diagnostics
        self._compiled_kernels = compiled_kernels
        self._backend_name = backend_name
        self._node_backend_names = node_backend_names
        self._source_request_periods = source_request_periods

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """compile時に生成したwarningを返す。"""

        return self._compile_diagnostics

    def run(self, *, duration: float | None = None) -> RunResult:
        """単一threadの決定的SchedulerでPlanを実行する。

        Args:
            duration: Sourceの論理時間上限。Noneではfinite SourceのEOFまで実行。

        Returns:
            collector結果、Diagnostic、status件数を持つRunResult。
        """

        runtime = _PlanRuntime(self, duration)
        return runtime.run()

    def export(self, path: str | Path) -> None:
        """required Node、output、compile DiagnosticをJSONまたはDOTへ出力する。

        Raises:
            ValueError: 拡張子が`.json`または`.dot`でない場合。
        """

        output_path = Path(path)
        if output_path.suffix == ".json":
            payload = {
                "schema_version": "0.1",
                "kind": "execution_plan",
                "backend": self._backend_name,
                "source_request_periods": {
                    str(node_id): str(period)
                    for node_id, period in self._source_request_periods.items()
                },
                "nodes": [
                    {
                        "id": node.id,
                        "kind": node.kind.value,
                        "output_port": node.output_port,
                        "config_scope_id": node.config.scope_id,
                        "rate_period": str(node.rate_period) if node.rate_period else None,
                        "execution_domain": self._node_backend_names[node.id],
                    }
                    for node in self._nodes
                ],
                "outputs": [
                    {
                        "index": index,
                        "port_id": item.flow.port_id,
                        "collector": type(item.collector).__name__,
                    }
                    for index, item in enumerate(self._outputs)
                ],
                "diagnostics": [
                    {
                        "severity": item.severity.value,
                        "code": item.code,
                        "message": item.message,
                        "node_id": item.node_id,
                        "port_id": item.port_id,
                    }
                    for item in self._compile_diagnostics
                ],
            }
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
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


class _PlanRuntime:
    def __init__(self, plan: ExecutionPlan, duration: float | None) -> None:
        self.plan = plan
        self.duration = None if duration is None else Fraction(str(duration))
        self.nodes_by_port = {node.output_port: node for node in plan._nodes}
        self.nodes_by_id = {node.id: node for node in plan._nodes}
        self.consumers: dict[int, list[tuple[NodeSpec, int]]] = defaultdict(list)
        for node in plan._nodes:
            for index, item in enumerate(node.inputs):
                self.consumers[item.source_port].append((node, index))
        self.queues: dict[tuple[int, int], deque[Emission[object]]] = defaultdict(deque)
        self.latest: dict[tuple[int, int], Emission[object]] = {}
        self.frames: dict[int, _FrameState] = {}
        self.rates: dict[int, _RateState] = {}
        self.port_sequences: dict[int, int] = defaultdict(int)
        self.diagnostics = list(plan._compile_diagnostics)
        self.status_counts: Counter[EmissionStatus] = Counter()
        self.collectors: dict[int, CollectorSession[Any]] = {
            item.flow.port_id: item.collector.create_session() for item in plan._outputs
        }
        self.kernel_sessions: dict[int, CompiledKernelSession[object]] = {
            node_id: kernel.create_session() for node_id, kernel in plan._compiled_kernels.items()
        }
        self.output_ports = set(self.collectors)

    def run(self) -> RunResult:
        context = PlanContext(required_node_count=len(self.plan._nodes))
        initialized: list[Extension] = []
        try:
            for extension in self.plan._extensions:
                extension.initialize(context)
                initialized.append(extension)
            sources = [node for node in self.plan._nodes if node.kind is NodeKind.SOURCE]
            iterators = {node.id: iter(self._source_values(node)) for node in sources}
            active = {node.id for node in sources}
            source_indexes: dict[int, int] = defaultdict(int)

            while active:
                for node in sources:
                    if node.id not in active:
                        continue
                    try:
                        value = next(iterators[node.id])
                    except StopIteration:
                        active.remove(node.id)
                        continue
                    emission = self._source_emission(value, source_indexes[node.id])
                    source_indexes[node.id] += 1
                    if (
                        self.duration is not None
                        and emission.interval.end.as_fraction() > self.duration
                    ):
                        active.remove(node.id)
                        continue
                    self._publish(node.output_port, emission)
                    self._drain_ready_nodes()

            self._flush_padded_frames()
            self._drain_ready_nodes()
            self._report_unmatched_inputs()
            return self._result()
        finally:
            # collectorやKernelが失敗しても、Extensionが外部資源を閉じられるようにする。
            for extension in reversed(initialized):
                extension.finalize(context)

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
        for extension in self.plan._extensions:
            if port_id in extension.observed_ports():
                extension.on_output(event)
        if port_id in self.output_ports:
            self.collectors[port_id].add(emission)
        for node, input_index in self.consumers.get(port_id, ()):
            self.queues[(node.id, input_index)].append(emission)

    def _drain_ready_nodes(self) -> None:
        while True:
            progressed = False
            for node in self.plan._nodes:
                if node.kind is NodeKind.SOURCE:
                    continue
                if (
                    node.kind is NodeKind.FRAME
                    and self._process_frame_input(node)
                    or node.kind is NodeKind.RATE
                    and self._process_rate_input(node)
                    or node.kind is NodeKind.MAP
                    and self._process_map_if_ready(node)
                ):
                    progressed = True
            if not progressed:
                return

    def _process_frame_input(self, node: NodeSpec) -> bool:
        queue = self.queues[(node.id, 0)]
        if not queue:
            return False
        state = self.frames.setdefault(node.id, _FrameState([]))
        emission = queue.popleft()
        if state.skip_remaining:
            state.skip_remaining -= 1
            return True
        state.items.append(emission)
        if node.frame_size is None or node.frame_hop is None:
            raise RuntimeError("FRAME Node lacks size or hop")
        if len(state.items) < node.frame_size:
            return True
        self._emit_frame(node, state.items[: node.frame_size])
        if node.frame_hop <= len(state.items):
            del state.items[: node.frame_hop]
        else:
            state.items.clear()
            state.skip_remaining = node.frame_hop - node.frame_size
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

    def _flush_padded_frames(self) -> None:
        for node in self.plan._nodes:
            if node.kind is not NodeKind.FRAME or not node.pad_end:
                continue
            state = self.frames.get(node.id)
            if state is None or not state.items or node.frame_size is None:
                continue
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
        for emission in self._normalize_result(
            result,
            main.interval,
            node.output_port,
            inherited_status,
            inherited_diagnostics,
        ):
            self._publish(node.output_port, emission)
        return True

    def _next_sequence(self, port_id: int) -> int:
        sequence = self.port_sequences[port_id]
        self.port_sequences[port_id] += 1
        return sequence

    def _normalize_result(
        self,
        result: object,
        interval: LogicalInterval,
        port_id: int,
        inherited_status: EmissionStatus,
        inherited_diagnostics: tuple[Diagnostic, ...],
    ) -> tuple[Emission[object], ...]:
        if isinstance(result, Skip):
            return ()
        values = result.values if isinstance(result, EmitMany) else (result,)
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
            for extension in self.plan._extensions:
                extension.on_diagnostic(diagnostic)

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
