"""runtimeを横断観測するExtension protocolとSnapshotを提供する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .graph import Flow
from .model import Diagnostic, Emission, EmissionStatus


@dataclass(frozen=True)
class PlanContext:
    """Extensionへ公開する一回のrunの基本情報。"""

    required_node_count: int


@dataclass(frozen=True)
class OutputEvent:
    """一つのPortからEmissionが出たことを表す。"""

    port_id: int
    emission: Emission[object]


class Trigger(Protocol):
    """Extension callbackを呼ぶ時点を決めるprotocol。"""

    def should_fire(self, event_index: int) -> bool:
        """0始まりevent indexでcallbackを呼ぶか返す。"""

        ...


@dataclass(frozen=True)
class Always:
    """すべての対象eventで発火するTrigger。"""

    def should_fire(self, event_index: int) -> bool:
        """常にTrueを返す。"""

        return True


@dataclass(frozen=True)
class Every:
    """一定event数ごとに発火するTrigger。"""

    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("trigger count must be positive")

    def should_fire(self, event_index: int) -> bool:
        """count件ごとのeventでTrueを返す。"""

        return event_index % self.count == 0


class Extension(Protocol):
    """計算Nodeと分離した観測処理のprotocol。"""

    @property
    def priority(self) -> int:
        """同一event内の決定的な呼出順を返す。"""

        ...

    @property
    def trigger(self) -> Trigger:
        """Extension callbackの発火条件を返す。"""

        ...

    def initialize(self, context: PlanContext) -> None:
        """run開始前に一回呼ばれる。"""

    def on_output(self, event: OutputEvent) -> None:
        """対象PortのEmissionを観測する。"""

    def on_diagnostic(self, diagnostic: Diagnostic) -> None:
        """runtime Diagnosticを観測する。"""

    def finalize(self, context: PlanContext) -> None:
        """run終了時に一回呼ばれる。"""

    def observed_ports(self) -> tuple[int, ...]:
        """ExecutionPlanへ含める観測Portを返す。"""

        ...


class Snapshot:
    """指定FlowのEmissionをJSON Linesへ同期保存するExtension。

    v0.1ではbounded queueを持たずScheduler threadで書き込むため、保存完了と
    計算完了の順序が一致する。任意値はJSON不能でもreprとして欠落なく残す。
    """

    priority = 0
    trigger: Trigger

    def __init__(
        self,
        *,
        flow: Flow[Any],
        path: str | Path,
        include_degraded: bool = True,
        trigger: Trigger | None = None,
    ) -> None:
        self._port_id = flow.port_id
        self._path = Path(path)
        self._include_degraded = include_degraded
        self.trigger = Always() if trigger is None else trigger
        self._event_index = 0

    def observed_ports(self) -> tuple[int, ...]:
        """Snapshot対象Portを一件返す。"""

        return (self._port_id,)

    def initialize(self, context: PlanContext) -> None:
        """出力directoryを作成し、前回runのfileを切り詰める。"""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")
        self._event_index = 0

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, Mapping):
            return dict(value)
        return repr(value)

    def on_output(self, event: OutputEvent) -> None:
        """対象PortのEmissionをstatusとDiagnostic付きで追記する。"""

        if event.port_id != self._port_id:
            return
        index = self._event_index
        self._event_index += 1
        if not self.trigger.should_fire(index):
            return
        if not self._include_degraded and event.emission.status is not EmissionStatus.OK:
            return
        payload = {
            "sequence": event.emission.sequence,
            "status": event.emission.status.value,
            "interval": {
                "start": str(event.emission.interval.start.as_fraction()),
                "end": str(event.emission.interval.end.as_fraction()),
            },
            "value": event.emission.value,
            "diagnostics": [item.code for item in event.emission.diagnostics],
        }
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=self._json_default))
            stream.write("\n")

    def on_diagnostic(self, diagnostic: Diagnostic) -> None:
        """DiagnosticはEmission側に保存するため、このExtensionでは追加処理しない。"""

    def finalize(self, context: PlanContext) -> None:
        """同期書込みのため追加flushを行わない。"""
