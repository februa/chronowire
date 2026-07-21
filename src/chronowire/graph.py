"""Flowから構築されるappend-only Logical Graphを定義する。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, TypeVar

from .config import Config
from .kernel import CallableAdapter, GapPolicy, Kernel
from .source import RealtimeSource, Source

T = TypeVar("T")
U = TypeVar("U")


class NodeKind(StrEnum):
    """v0.1 Logical GraphのNode種別を表す。"""

    SOURCE = "source"
    MAP = "map"
    FRAME = "frame"
    RATE = "rate"


class RatePolicy(StrEnum):
    """rate Nodeが発火時刻の値を選ぶ方法を表す。"""

    HOLD = "hold"


class InputSemantics(StrEnum):
    """Node入力の同期方法を表す。"""

    SYNCHRONOUS = "synchronous"
    LATEST = "latest"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    TOLERANCE = "tolerance"


class MissingInputPolicy(StrEnum):
    """同期入力が生成不能な場合のNode動作を表す。"""

    STALL = "stall"
    SKIP = "skip"


@dataclass(frozen=True)
class InputSpec:
    """一つのNode入力と、その同期意味論を表す。"""

    source_port: int
    semantics: InputSemantics
    keyword: str | None = None
    tolerance: Fraction | None = None
    missing_policy: MissingInputPolicy = MissingInputPolicy.STALL


@dataclass(frozen=True)
class NodeSpec:
    """Logical Graphへ登録する不変なNode定義を表す。"""

    id: int
    kind: NodeKind
    output_port: int
    output_ports: tuple[int, ...]
    inputs: tuple[InputSpec, ...]
    config: Config
    operation: Callable[..., object] | Kernel[object] | None = None
    constants: Mapping[str, object] | None = None
    config_paths: tuple[str, ...] | None = None
    accepts_invalid: bool = False
    source: Iterable[object] | Source[object] | RealtimeSource[object] | None = None
    frame_size: int | None = None
    frame_hop: int | None = None
    pad_end: bool = False
    rate_period: Fraction | None = None
    rate_policy: RatePolicy | None = None
    max_items: int = 1
    time_transform: str = "preserve"
    gap_policy: GapPolicy = GapPolicy.RESET


@dataclass(frozen=True)
class NodeInfo:
    """Graphの公開読み取り用Node情報を表す。"""

    id: int
    kind: NodeKind
    output_port: int
    output_ports: tuple[int, ...]
    input_ports: tuple[int, ...]
    config_scope_id: str
    rate_period: Fraction | None
    rate_policy: RatePolicy | None


@dataclass(frozen=True)
class EdgeInfo:
    """Graphの公開読み取り用Edge情報を表す。"""

    source_port: int
    target_node: int
    target_input: int
    semantics: InputSemantics
    keyword: str | None
    tolerance: Fraction | None
    missing_policy: MissingInputPolicy


@dataclass(frozen=True)
class GraphInfo:
    """Logical Graphの読み取り専用snapshotを表す。"""

    nodes: tuple[NodeInfo, ...]
    edges: tuple[EdgeInfo, ...]


class Graph:
    """NodeとPortをappend-onlyで保持するLogical Graph。

    runtime bufferやKernelStateは保持せず、compile前の論理構造だけを管理する。
    """

    def __init__(self) -> None:
        self._nodes: list[NodeSpec] = []
        self._port_to_node: dict[int, int] = {}

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        """登録順の不変Node列を返す。"""

        return tuple(self._nodes)

    def node_for_port(self, port_id: int) -> NodeSpec:
        """output Portを生成するNodeを返す。

        Raises:
            KeyError: PortがGraphに存在しない場合。
        """

        return self._nodes[self._port_to_node[port_id]]

    def add_node(
        self,
        kind: NodeKind,
        config: Config,
        *,
        inputs: tuple[InputSpec, ...] = (),
        operation: Callable[..., object] | Kernel[object] | None = None,
        constants: Mapping[str, object] | None = None,
        config_paths: tuple[str, ...] | None = None,
        accepts_invalid: bool = False,
        source: Iterable[object] | Source[object] | RealtimeSource[object] | None = None,
        frame_size: int | None = None,
        frame_hop: int | None = None,
        pad_end: bool = False,
        rate_period: Fraction | None = None,
        rate_policy: RatePolicy | None = None,
        max_items: int = 1,
        time_transform: str = "preserve",
        gap_policy: GapPolicy = GapPolicy.RESET,
    ) -> int:
        """単一output Nodeを追加し、新しいPort IDを返す。"""

        return self.add_node_ports(
            kind,
            config,
            inputs=inputs,
            operation=operation,
            constants=constants,
            config_paths=config_paths,
            accepts_invalid=accepts_invalid,
            source=source,
            frame_size=frame_size,
            frame_hop=frame_hop,
            pad_end=pad_end,
            rate_period=rate_period,
            rate_policy=rate_policy,
            max_items=max_items,
            time_transform=time_transform,
            gap_policy=gap_policy,
            output_count=1,
        )[0]

    def add_node_ports(
        self,
        kind: NodeKind,
        config: Config,
        *,
        inputs: tuple[InputSpec, ...] = (),
        operation: Callable[..., object] | Kernel[object] | None = None,
        constants: Mapping[str, object] | None = None,
        config_paths: tuple[str, ...] | None = None,
        accepts_invalid: bool = False,
        source: Iterable[object] | Source[object] | RealtimeSource[object] | None = None,
        frame_size: int | None = None,
        frame_hop: int | None = None,
        pad_end: bool = False,
        rate_period: Fraction | None = None,
        rate_policy: RatePolicy | None = None,
        max_items: int = 1,
        output_count: int,
        time_transform: str = "preserve",
        gap_policy: GapPolicy = GapPolicy.RESET,
    ) -> tuple[int, ...]:
        """固定数のoutput Portを持つNodeを追加する。"""

        if output_count <= 0:
            raise ValueError("output_count must be positive")

        node_id = len(self._nodes)
        first_port = len(self._port_to_node)
        output_ports = tuple(range(first_port, first_port + output_count))
        port_id = output_ports[0]
        node = NodeSpec(
            id=node_id,
            kind=kind,
            output_port=port_id,
            output_ports=output_ports,
            inputs=inputs,
            config=config,
            operation=operation,
            constants=constants,
            config_paths=config_paths,
            accepts_invalid=accepts_invalid,
            source=source,
            frame_size=frame_size,
            frame_hop=frame_hop,
            pad_end=pad_end,
            rate_period=rate_period,
            rate_policy=rate_policy,
            max_items=max_items,
            time_transform=time_transform,
            gap_policy=gap_policy,
        )
        self._nodes.append(node)
        for output_port in output_ports:
            self._port_to_node[output_port] = node_id
        return output_ports

    def info(self) -> GraphInfo:
        """利用者が変更できないGraphInfoを返す。"""

        nodes = tuple(
            NodeInfo(
                id=node.id,
                kind=node.kind,
                output_port=node.output_port,
                output_ports=node.output_ports,
                input_ports=tuple(item.source_port for item in node.inputs),
                config_scope_id=node.config.scope_id,
                rate_period=node.rate_period,
                rate_policy=node.rate_policy,
            )
            for node in self._nodes
        )
        edges = tuple(
            EdgeInfo(
                source_port=input_spec.source_port,
                target_node=node.id,
                target_input=input_index,
                semantics=input_spec.semantics,
                keyword=input_spec.keyword,
                tolerance=input_spec.tolerance,
                missing_policy=input_spec.missing_policy,
            )
            for node in self._nodes
            for input_index, input_spec in enumerate(node.inputs)
        )
        return GraphInfo(nodes=nodes, edges=edges)

    def export(self, path: str | Path) -> None:
        """Logical Graph全体をJSONまたはDOTへ出力する。

        Raises:
            ValueError: 拡張子が`.json`または`.dot`でない場合。
        """

        output_path = Path(path)
        info = self.info()
        if output_path.suffix == ".json":
            payload = {
                "schema_version": "0.1",
                "kind": "logical_graph",
                "nodes": [
                    {
                        "id": node.id,
                        "kind": node.kind.value,
                        "output_port": node.output_port,
                        "output_ports": node.output_ports,
                        "input_ports": node.input_ports,
                        "config_scope_id": node.config_scope_id,
                        "rate_period": str(node.rate_period) if node.rate_period else None,
                        "rate_policy": node.rate_policy.value if node.rate_policy else None,
                    }
                    for node in info.nodes
                ],
                "edges": [
                    {
                        "source_port": edge.source_port,
                        "target_node": edge.target_node,
                        "target_input": edge.target_input,
                        "semantics": edge.semantics.value,
                        "keyword": edge.keyword,
                        "tolerance": str(edge.tolerance) if edge.tolerance is not None else None,
                        "missing_policy": edge.missing_policy.value,
                    }
                    for edge in info.edges
                ],
            }
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if output_path.suffix == ".dot":
            lines = ["digraph chronowire {"]
            lines.extend(
                f'  n{node.id} [label="{node.id}: {node.kind.value}"];' for node in info.nodes
            )
            lines.extend(
                f"  n{self._port_to_node[edge.source_port]} -> n{edge.target_node} "
                f'[label="{edge.semantics.value}"];'
                for edge in info.edges
            )
            lines.append("}")
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        raise ValueError("Graph export supports only .json and .dot")


class StateFlow(Generic[T]):
    """指定Flowをlatest state入力として参照する公開handle。"""

    __slots__ = ("flow",)

    def __init__(self, flow: Flow[T]) -> None:
        self.flow = flow


class SynchronizedFlow(Generic[T]):
    """追加Flow入力へ明示したinterval同期契約を保持するhandle。"""

    __slots__ = ("flow", "semantics", "tolerance", "missing_policy")

    def __init__(
        self,
        flow: Flow[T],
        semantics: InputSemantics,
        tolerance: Fraction | None,
        missing_policy: MissingInputPolicy,
    ) -> None:
        self.flow = flow
        self.semantics = semantics
        self.tolerance = tolerance
        self.missing_policy = missing_policy


class Flow(Generic[T]):
    """Logical Graph上の一つのoutput Portを指す公開handle。

    Flow自身は値、runtime buffer、実行結果を保持しない。
    """

    __slots__ = ("_config", "_graph", "_port_id")

    def __init__(
        self,
        source: Iterable[T] | Source[T] | RealtimeSource[T],
        config: Config | None = None,
    ) -> None:
        self._graph = Graph()
        self._config = config if config is not None else Config()
        self._port_id = self._graph.add_node(
            NodeKind.SOURCE,
            self._config,
            source=source,
        )

    @classmethod
    def _from_port(cls, graph: Graph, port_id: int, config: Config) -> Flow[Any]:
        instance = cls.__new__(cls)
        instance._graph = graph
        instance._port_id = port_id
        instance._config = config
        return instance

    @property
    def port_id(self) -> int:
        """このFlowが参照するoutput Port IDを返す。"""

        return self._port_id

    @property
    def config(self) -> Config:
        """後段Nodeが参照する現在のConfig scopeを返す。"""

        return self._config

    def with_config(self, config: Config) -> Flow[T]:
        """同じPortを参照し、後段だけに別Config scopeを適用するFlowを返す。"""

        return self._from_port(self._graph, self._port_id, config)

    def latest(self) -> StateFlow[T]:
        """このFlowをlatest state入力として渡すhandleを返す。"""

        return StateFlow(self)

    def state_source(
        self,
        source: Iterable[U] | Source[U] | RealtimeSource[U],
        *,
        config: Config | None = None,
    ) -> StateFlow[U]:
        """同じGraphへ外部制御値Sourceを追加しlatest StateFlowを返す。

        Args:
            source: iterable、pull Source、またはRealtime Source。
            config: 制御Source固有の不変Config。Noneでは現在scopeを使用。

        Returns:
            後段MAPのkeyword引数へ渡せるlatest state handle。

        境界条件:
            制御値はConfigへ書き込まず、独立SOURCE PortとLATEST Edgeとして記録する。
        """

        state_config = self._config if config is None else config
        port_id = self._graph.add_node(NodeKind.SOURCE, state_config, source=source)
        return StateFlow(self._from_port(self._graph, port_id, state_config))

    def synchronize(
        self,
        semantics: InputSemantics,
        *,
        tolerance: int | float | Fraction | None = None,
        missing: MissingInputPolicy = MissingInputPolicy.STALL,
    ) -> SynchronizedFlow[T]:
        """このFlowを追加入力として使うinterval同期契約を返す。

        Args:
            semantics: CONTAINS、OVERLAPS、TOLERANCEのいずれか。
            tolerance: TOLERANCEで許容する両端時刻差。非負値。
            missing: 適合入力が生成不能な場合のSTALLまたはSKIP。

        Returns:
            主入力をreferenceとする同期handle。

        Raises:
            ValueError: semantics、tolerance、missingの組合せが不正な場合。
        """

        if semantics not in {
            InputSemantics.CONTAINS,
            InputSemantics.OVERLAPS,
            InputSemantics.TOLERANCE,
        }:
            raise ValueError("synchronize requires contains, overlaps, or tolerance semantics")
        resolved: Fraction | None = None
        if tolerance is not None:
            resolved = (
                Fraction(str(tolerance)) if isinstance(tolerance, float) else Fraction(tolerance)
            )
            if resolved < 0:
                raise ValueError("synchronization tolerance must not be negative")
        if semantics is InputSemantics.TOLERANCE and resolved is None:
            raise ValueError("tolerance semantics requires an explicit tolerance")
        if semantics is not InputSemantics.TOLERANCE and resolved is not None:
            raise ValueError("tolerance is valid only for tolerance semantics")
        if not isinstance(missing, MissingInputPolicy):
            raise ValueError("missing must be a MissingInputPolicy")
        return SynchronizedFlow(self, semantics, resolved, missing)

    def map(
        self,
        operation: Callable[..., U] | Kernel[U],
        *,
        config_paths: tuple[str, ...] | None = None,
        accepts_invalid: bool = False,
        max_items: int = 1,
        **arguments: object,
    ) -> Flow[U]:
        """Python callableをMAP NodeとしてGraphへ登録する。

        Args:
            operation: 主入力値と追加引数を受け取る処理。
            config_paths: callableが参照するConfig属性path。Noneはscope全体への依存。
            accepts_invalid: INVALID入力でもcallableを実行する場合にTrue。
            max_items: 一回のKernel呼出しが生成できるEmission上限。
            arguments: 定数、同期Flow、またはlatest StateFlow。

        Raises:
            ValueError: 異なるGraphのFlowを入力した場合、またはmax_itemsが正でない場合。
        """

        if isinstance(operation, CallableAdapter):
            if max_items != 1 or accepts_invalid:
                raise ValueError(
                    "CallableAdapter max_items/accepts_invalid must not be overridden in map"
                )
            max_items = operation.max_items
            accepts_invalid = operation.accepts_invalid
            time_transform = operation.time_transform
            gap_policy = operation.gap_policy
        else:
            time_transform = "preserve"
            gap_policy = GapPolicy.RESET
        if max_items <= 0:
            raise ValueError("map max_items must be positive")

        inputs = [InputSpec(self._port_id, InputSemantics.SYNCHRONOUS)]
        constants: dict[str, object] = {}
        for name, value in arguments.items():
            if isinstance(value, StateFlow):
                other = value.flow
                semantics = InputSemantics.LATEST
                tolerance = None
                missing_policy = MissingInputPolicy.STALL
            elif isinstance(value, SynchronizedFlow):
                other = value.flow
                semantics = value.semantics
                tolerance = value.tolerance
                missing_policy = value.missing_policy
            elif isinstance(value, Flow):
                other = value
                semantics = InputSemantics.SYNCHRONOUS
                tolerance = None
                missing_policy = MissingInputPolicy.STALL
            else:
                constants[name] = value
                continue
            if other._graph is not self._graph:
                raise ValueError(f"Flow argument {name!r} belongs to a different Graph")
            inputs.append(
                InputSpec(
                    other.port_id,
                    semantics,
                    keyword=name,
                    tolerance=tolerance,
                    missing_policy=missing_policy,
                )
            )

        port_id = self._graph.add_node(
            NodeKind.MAP,
            self._config,
            inputs=tuple(inputs),
            operation=operation,
            constants=constants,
            config_paths=config_paths,
            accepts_invalid=accepts_invalid,
            max_items=max_items,
            time_transform=time_transform,
            gap_policy=gap_policy,
        )
        return self._from_port(self._graph, port_id, self._config)

    def map_outputs(
        self,
        operation: Callable[..., object] | Kernel[object],
        *,
        output_count: int,
        config_paths: tuple[str, ...] | None = None,
        accepts_invalid: bool = False,
        max_items: int = 1,
        **arguments: object,
    ) -> tuple[Flow[Any], ...]:
        """明示数の複数output Portを持つMAP Nodeを登録する。

        Args:
            operation: `KernelOutputs`を返す処理。
            output_count: 固定output Port数。2以上。
            config_paths: callableが参照するConfig属性path。
            accepts_invalid: INVALID入力を実行する場合にTrue。
            max_items: 各Portの一回あたりEmission上限。
            arguments: 定数または追加Flow入力。

        Returns:
            output index順の通常Flow handle tuple。

        Raises:
            ValueError: output_count、Graph、max_itemsが不正な場合。
        """

        if output_count < 2:
            raise ValueError("map_outputs output_count must be at least two")
        if isinstance(operation, CallableAdapter):
            if max_items != 1 or accepts_invalid:
                raise ValueError("CallableAdapter max_items/accepts_invalid must not be overridden")
            max_items = operation.max_items
            accepts_invalid = operation.accepts_invalid
            time_transform = operation.time_transform
            gap_policy = operation.gap_policy
        else:
            time_transform = "preserve"
            gap_policy = GapPolicy.RESET
        if max_items <= 0:
            raise ValueError("map_outputs max_items must be positive")
        inputs = [InputSpec(self._port_id, InputSemantics.SYNCHRONOUS)]
        constants: dict[str, object] = {}
        for name, value in arguments.items():
            if isinstance(value, StateFlow):
                other, semantics = value.flow, InputSemantics.LATEST
                tolerance, missing_policy = None, MissingInputPolicy.STALL
            elif isinstance(value, SynchronizedFlow):
                other, semantics = value.flow, value.semantics
                tolerance, missing_policy = value.tolerance, value.missing_policy
            elif isinstance(value, Flow):
                other, semantics = value, InputSemantics.SYNCHRONOUS
                tolerance, missing_policy = None, MissingInputPolicy.STALL
            else:
                constants[name] = value
                continue
            if other._graph is not self._graph:
                raise ValueError(f"Flow argument {name!r} belongs to a different Graph")
            inputs.append(InputSpec(other.port_id, semantics, name, tolerance, missing_policy))
        ports = self._graph.add_node_ports(
            NodeKind.MAP,
            self._config,
            inputs=tuple(inputs),
            operation=operation,
            constants=constants,
            config_paths=config_paths,
            accepts_invalid=accepts_invalid,
            max_items=max_items,
            output_count=output_count,
            time_transform=time_transform,
            gap_policy=gap_policy,
        )
        return tuple(self._from_port(self._graph, port, self._config) for port in ports)

    def frame(
        self,
        size: int,
        *,
        hop: int | None = None,
        pad_end: bool = False,
    ) -> Flow[tuple[T | None, ...]]:
        """入力を固定長frameへ蓄積するNodeを登録する。

        Args:
            size: 一つのframeに含めるitem数。正の整数。
            hop: 隣接frame間の開始item差。Noneではsize。
            pad_end: EOF時の不足frameをNoneでpaddingする場合にTrue。

        Raises:
            ValueError: sizeまたはhopが正でない場合。
        """

        resolved_hop = size if hop is None else hop
        if size <= 0 or resolved_hop <= 0:
            raise ValueError("frame size and hop must be positive")
        port_id = self._graph.add_node(
            NodeKind.FRAME,
            self._config,
            inputs=(InputSpec(self._port_id, InputSemantics.SYNCHRONOUS),),
            frame_size=size,
            frame_hop=resolved_hop,
            pad_end=pad_end,
        )
        return self._from_port(self._graph, port_id, self._config)

    def rate(
        self,
        frequency_hz: int | float | Fraction,
        *,
        policy: RatePolicy = RatePolicy.HOLD,
    ) -> Flow[T]:
        """入力値を指定した論理周期で発火するRATE Nodeを登録する。

        Args:
            frequency_hz: 1秒相当の論理時間あたりの発火回数。正数。
            policy: 発火境界で使う値の選択規則。v0.1はHOLDだけを扱う。

        Returns:
            各発火区間に、その時点の入力値を保持して出力するFlow。

        Raises:
            ValueError: frequency_hzが正の有限値でない場合。

        境界条件:
            数値補間やアンチエイリアス処理は行わない。入力interval内にある発火時刻へ
            同じ値を割り当て、出力intervalを正確な有理周期で表す。
        """

        try:
            frequency = (
                Fraction(str(frequency_hz))
                if isinstance(frequency_hz, float)
                else Fraction(frequency_hz)
            )
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("rate frequency must be a positive finite value") from error
        if frequency <= 0:
            raise ValueError("rate frequency must be positive")
        if policy is not RatePolicy.HOLD:
            raise ValueError(f"unsupported rate policy {policy!r}")
        port_id = self._graph.add_node(
            NodeKind.RATE,
            self._config,
            inputs=(InputSpec(self._port_id, InputSemantics.SYNCHRONOUS),),
            rate_period=Fraction(1, 1) / frequency,
            rate_policy=policy,
        )
        return self._from_port(self._graph, port_id, self._config)

    def graph_info(self) -> GraphInfo:
        """このFlowが属するLogical Graph全体のsnapshotを返す。"""

        return self._graph.info()

    def export(self, path: str | Path) -> None:
        """このFlowが属するLogical Graph全体を出力する。"""

        self._graph.export(path)
