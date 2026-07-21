"""compile-time観測契約とrun-local Extension handlerを定義する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .graph import Flow
from .model import Diagnostic, Emission, EmissionStatus

EXTENSION_ABI_VERSION = "chronowire.extension.v1"


@dataclass(frozen=True)
class PlanContext:
    """Extensionへ公開する一回のrunの基本情報。"""

    required_node_count: int


@dataclass(frozen=True)
class OutputEvent:
    """一つのPortからEmissionが出たことを表す。"""

    port_id: int
    emission: Emission[object]


class TriggerSession(Protocol):
    """一回のrunに閉じたtrigger状態のprotocol。"""

    def should_fire(self, emission: Emission[object]) -> bool:
        """対象Emissionでhandlerを呼ぶ場合にTrueを返す。"""

        ...


class Trigger(Protocol):
    """compile時に固定するExtension発火条件のprotocol。"""

    @property
    def kind(self) -> str:
        """PortablePlanIRへ記録するtrigger種別を返す。"""

        ...

    def create_session(self) -> TriggerSession:
        """空のrun-local trigger状態を生成する。"""

        ...


class _AlwaysSession:
    def should_fire(self, emission: Emission[object]) -> bool:
        return True


@dataclass(frozen=True)
class Always:
    """すべての対象Emissionで発火するTrigger。"""

    kind = "always"

    def create_session(self) -> TriggerSession:
        """状態を持たないAlways sessionを返す。"""

        return _AlwaysSession()


class _EverySession:
    def __init__(self, count: int) -> None:
        self._count = count
        self._event_index = 0

    def should_fire(self, emission: Emission[object]) -> bool:
        result = self._event_index % self._count == 0
        self._event_index += 1
        return result


@dataclass(frozen=True)
class Every:
    """対象Emissionを一定件数ごとに観測するTrigger。"""

    count: int
    kind = "every_event"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("trigger count must be positive")

    def create_session(self) -> TriggerSession:
        """event indexを0から数えるrun-local sessionを返す。"""

        return _EverySession(self.count)


def _fraction(value: int | float | Fraction, field: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite rational number")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} must be a finite rational number") from error
    return result


class _EveryLogicalTimeSession:
    def __init__(self, period: Fraction, phase: Fraction) -> None:
        self._period = period
        self._next_boundary = phase

    def should_fire(self, emission: Emission[object]) -> bool:
        start = emission.interval.start.as_fraction()
        end = emission.interval.end.as_fraction()
        if self._next_boundary < start:
            skipped = -(-(start - self._next_boundary) // self._period)
            self._next_boundary += skipped * self._period
        if not start <= self._next_boundary < end:
            return False
        while self._next_boundary < end:
            self._next_boundary += self._period
        return True


@dataclass(frozen=True)
class EveryLogicalTime:
    """論理時間境界を含むEmissionを一定周期で観測するTrigger。

    境界は半開区間`[interval.start, interval.end)`へ所属する。同じ境界を
    複数Emissionが覆う場合も、一つの観測契約では最初のEmissionだけが発火する。

    Args:
        period: 正の論理時間周期。
        phase: 最初の境界。periodで正規化する。

    Raises:
        ValueError: periodが正でない、または有限な有理数へ変換できない場合。
    """

    period: Fraction
    phase: Fraction = Fraction(0)
    kind = "every_logical_time"

    def __init__(
        self,
        period: int | float | Fraction,
        phase: int | float | Fraction = 0,
    ) -> None:
        normalized_period = _fraction(period, "trigger period")
        normalized_phase = _fraction(phase, "trigger phase")
        if normalized_period <= 0:
            raise ValueError("trigger period must be positive")
        object.__setattr__(self, "period", normalized_period)
        object.__setattr__(self, "phase", normalized_phase % normalized_period)

    def create_session(self) -> TriggerSession:
        """次の論理境界を保持するrun-local sessionを返す。"""

        return _EveryLogicalTimeSession(self.period, self.phase)


class ExtensionFailurePolicy(StrEnum):
    """Extension handler失敗時のv0.1 policyを表す。"""

    FAIL = "fail"


class ExtensionOverflowPolicy(StrEnum):
    """Extension内部queue上限到達時のv0.1 policyを表す。"""

    FAIL = "fail"


@dataclass(frozen=True)
class ObservationSpec:
    """compile時に固定する一つのPort観測契約。"""

    flow: Flow[Any]
    extension_id: str
    trigger: Trigger
    priority: int
    failure_policy: ExtensionFailurePolicy
    overflow_policy: ExtensionOverflowPolicy
    abi_version: str = EXTENSION_ABI_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.extension_id, str) or not self.extension_id.strip():
            raise ValueError("extension_id must be a non-empty string")
        if not isinstance(self.abi_version, str) or not self.abi_version:
            raise ValueError("extension ABI version must be non-empty")


def observe(
    flow: Flow[Any],
    *,
    extension_id: str,
    trigger: Trigger | None = None,
    priority: int = 0,
    failure_policy: ExtensionFailurePolicy = ExtensionFailurePolicy.FAIL,
    overflow_policy: ExtensionOverflowPolicy = ExtensionOverflowPolicy.FAIL,
) -> ObservationSpec:
    """Flowから不変なcompile-time観測契約を生成する。

    Args:
        flow: 観測対象Portを指すFlow。
        extension_id: Plan内で一意な利用者指定の安定ID。
        trigger: Emission発火条件。NoneではAlways。
        priority: 同一Emissionで小さい値から呼ぶ決定的優先順位。
        failure_policy: handler失敗時policy。v0.1はFAILのみ。
        overflow_policy: handler内部queueのpolicy。v0.1はFAILのみ。

    Returns:
        compileへ渡すObservationSpec。

    Raises:
        ValueError: extension_idが空の場合。
    """

    if not isinstance(extension_id, str) or not extension_id.strip():
        raise ValueError("extension_id must be a non-empty string")
    return ObservationSpec(
        flow,
        extension_id,
        Always() if trigger is None else trigger,
        priority,
        failure_policy,
        overflow_policy,
    )


@runtime_checkable
class ExtensionSession(Protocol):
    """一回のrunに閉じたExtension handler状態のprotocol。"""

    def initialize(self, context: PlanContext) -> None:
        """run開始前に一回呼ばれる。"""

        ...

    def on_output(self, event: OutputEvent) -> None:
        """triggerが発火した対象PortのEmissionを観測する。"""

        ...

    def on_diagnostic(self, diagnostic: Diagnostic) -> None:
        """runtime Diagnosticを観測する。"""

        ...

    def finalize(self, context: PlanContext) -> None:
        """run終了時に一回呼ばれる。"""

        ...


@runtime_checkable
class Extension(Protocol):
    """runごとに独立したExtensionSessionを生成するbinding protocol。"""

    @property
    def abi_version(self) -> str:
        """ObservationSpecが要求するhandler ABI versionを返す。"""

        ...

    def create_session(self) -> ExtensionSession:
        """空のrun-local handler sessionを生成する。"""

        ...


class _SnapshotSession:
    def __init__(self, path: Path, include_degraded: bool) -> None:
        self._path = path
        self._include_degraded = include_degraded

    def initialize(self, context: PlanContext) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, Mapping):
            return dict(value)
        return repr(value)

    def on_output(self, event: OutputEvent) -> None:
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
        return

    def finalize(self, context: PlanContext) -> None:
        return


@dataclass(frozen=True)
class Snapshot:
    """EmissionをJSON Linesへ同期保存するExtension binding。

    Flow、extension_id、triggerは`observe()`側へ宣言し、このbindingには
    process-localな保存先と保存policyだけを持たせる。
    """

    path: str | Path
    include_degraded: bool = True

    @property
    def abi_version(self) -> str:
        """Snapshotが実装するExtension ABI versionを返す。"""

        return EXTENSION_ABI_VERSION

    def create_session(self) -> ExtensionSession:
        """前回runの状態を持たないSnapshot sessionを生成する。"""

        return _SnapshotSession(Path(self.path), self.include_degraded)
