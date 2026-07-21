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
from .kernel import Kernel
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


@dataclass(frozen=True)
class InputSpec:
    """一つのNode入力と、その同期意味論を表す。"""

    source_port: int
    semantics: InputSemantics
    keyword: str | None = None


@dataclass(frozen=True)
class NodeSpec:
    """Logical Graphへ登録する不変なNode定義を表す。"""

    id: int
    kind: NodeKind
    output_port: int
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


@dataclass(frozen=True)
class NodeInfo:
    """Graphの公開読み取り用Node情報を表す。"""

    id: int
    kind: NodeKind
    output_port: int
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
    ) -> int:
        """Nodeを追加し、新しいoutput Port IDを返す。"""

        node_id = len(self._nodes)
        port_id = node_id
        node = NodeSpec(
            id=node_id,
            kind=kind,
            output_port=port_id,
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
        )
        self._nodes.append(node)
        self._port_to_node[port_id] = node_id
        return port_id

    def info(self) -> GraphInfo:
        """利用者が変更できないGraphInfoを返す。"""

        nodes = tuple(
            NodeInfo(
                id=node.id,
                kind=node.kind,
                output_port=node.output_port,
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

        if max_items <= 0:
            raise ValueError("map max_items must be positive")

        inputs = [InputSpec(self._port_id, InputSemantics.SYNCHRONOUS)]
        constants: dict[str, object] = {}
        for name, value in arguments.items():
            if isinstance(value, StateFlow):
                other = value.flow
                semantics = InputSemantics.LATEST
            elif isinstance(value, Flow):
                other = value
                semantics = InputSemantics.SYNCHRONOUS
            else:
                constants[name] = value
                continue
            if other._graph is not self._graph:
                raise ValueError(f"Flow argument {name!r} belongs to a different Graph")
            inputs.append(InputSpec(other.port_id, semantics, keyword=name))

        port_id = self._graph.add_node(
            NodeKind.MAP,
            self._config,
            inputs=tuple(inputs),
            operation=operation,
            constants=constants,
            config_paths=config_paths,
            accepts_invalid=accepts_invalid,
            max_items=max_items,
        )
        return self._from_port(self._graph, port_id, self._config)

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
