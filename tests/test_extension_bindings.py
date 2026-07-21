"""compile-time観測契約とrun-local Extension bindingを検証する。"""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import chronowire as cw


class _RecordingSession:
    def __init__(self, values: list[object], label: str | None = None) -> None:
        self._values = values
        self._label = label

    def initialize(self, context: cw.PlanContext) -> None:
        return

    def on_output(self, event: cw.OutputEvent) -> None:
        self._values.append(event.emission.value if self._label is None else self._label)

    def on_diagnostic(self, diagnostic: cw.Diagnostic) -> None:
        return

    def finalize(self, context: cw.PlanContext) -> None:
        return


class _RecordingExtension:
    abi_version = "chronowire.extension.v1"

    def __init__(self, *, trace: list[object] | None = None, label: str | None = None) -> None:
        self._trace = trace
        self._label = label
        self.sessions: list[list[object]] = []

    def create_session(self) -> cw.ExtensionSession:
        values = self._trace if self._trace is not None else []
        if self._trace is None:
            self.sessions.append(values)
        return _RecordingSession(values, self._label)


@dataclass(frozen=True)
class _WrongAbiExtension:
    abi_version: str = "wrong.extension.v9"

    def create_session(self) -> cw.ExtensionSession:
        return _RecordingSession([])


class _FailingSession(_RecordingSession):
    def on_output(self, event: cw.OutputEvent) -> None:
        raise RuntimeError("recorder unavailable")


class _FailingExtension:
    abi_version = "chronowire.extension.v1"

    def create_session(self) -> cw.ExtensionSession:
        return _FailingSession([])


def test_every_logical_time_uses_half_open_interval_boundaries() -> None:
    """区間endと一致する境界は次のEmissionへ一意に所属する。"""

    source = cw.Flow(range(5))
    observation = cw.observe(
        source,
        extension_id="logical_snapshot",
        trigger=cw.EveryLogicalTime(period=2),
    )
    binding = _RecordingExtension()
    plan = cw.compile([source], extensions=[observation])

    plan.create_session(extension_bindings={"logical_snapshot": binding}).run()

    assert binding.sessions == [[0, 2, 4]]


def test_logical_trigger_state_is_reset_for_each_run() -> None:
    """同じExecutionSessionを再実行してもtrigger状態を持ち越さない。"""

    source = cw.Flow(range(4))
    observation = cw.observe(
        source,
        extension_id="periodic",
        trigger=cw.EveryLogicalTime(Fraction(2)),
    )
    binding = _RecordingExtension()
    session = cw.compile([source], extensions=[observation]).create_session(
        extension_bindings={"periodic": binding}
    )

    session.run()
    session.run()

    assert binding.sessions == [[0, 2], [0, 2]]


def test_extension_priority_then_registration_order_is_deterministic() -> None:
    """同一Portのhandlerをpriority昇順、同値なら登録順で呼ぶ。"""

    trace: list[object] = []
    source = cw.Flow([1])
    observations = [
        cw.observe(source, extension_id="late", priority=10),
        cw.observe(source, extension_id="early_first", priority=-1),
        cw.observe(source, extension_id="early_second", priority=-1),
    ]
    bindings = {
        "late": _RecordingExtension(trace=trace, label="late"),
        "early_first": _RecordingExtension(trace=trace, label="early_first"),
        "early_second": _RecordingExtension(trace=trace, label="early_second"),
    }

    cw.compile([source], extensions=observations).create_session(extension_bindings=bindings).run()

    assert trace == ["early_first", "early_second", "late"]


def test_duplicate_extension_id_is_compile_error() -> None:
    """安定ID重複をbinding時まで遅延せずcompileで拒否する。"""

    source = cw.Flow([1])
    observations = [
        cw.observe(source, extension_id="duplicate"),
        cw.observe(source, extension_id="duplicate"),
    ]

    with pytest.raises(cw.DuplicateExtensionIdError, match="ports 0 and 0"):
        cw.compile([source], extensions=observations)


def test_compile_rejects_runtime_handler_in_observation_list() -> None:
    """旧来のhandler直接指定を曖昧なAttributeErrorにせず責務付きで拒否する。"""

    source = cw.Flow([1])
    invalid_observation: Any = _RecordingExtension()

    with pytest.raises(cw.CompileError, match="ObservationSpec values returned by observe"):
        cw.compile([source], extensions=[invalid_observation])


def test_create_session_rejects_missing_unknown_type_and_abi_bindings() -> None:
    """Extension binding集合とABIをcreate_session時に完全検証する。"""

    source = cw.Flow([1])
    plan = cw.compile(
        [source],
        extensions=[cw.observe(source, extension_id="required")],
    )

    with pytest.raises(cw.ExtensionBindingError, match="missing required binding"):
        plan.create_session()
    with pytest.raises(cw.ExtensionBindingError, match="unknown or unused"):
        plan.create_session(
            extension_bindings={
                "required": _RecordingExtension(),
                "unused": _RecordingExtension(),
            }
        )
    invalid_binding: Any = object()
    with pytest.raises(cw.ExtensionBindingError, match="invalid binding type"):
        plan.create_session(extension_bindings={"required": invalid_binding})
    with pytest.raises(cw.ExtensionBindingError, match="does not match required"):
        plan.create_session(extension_bindings={"required": _WrongAbiExtension()})


def test_snapshot_records_degraded_and_invalid_without_entering_plan_ir(tmp_path: Path) -> None:
    """劣化値を保存しつつpathやhandler objectをPortablePlanIRへ含めない。"""

    degraded = cw.Emission(
        1,
        cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
        0,
        cw.EmissionStatus.DEGRADED,
    )
    invalid = cw.Emission(
        2,
        cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(2)),
        1,
        cw.EmissionStatus.INVALID,
    )
    source = cw.Flow([degraded, invalid])
    observation = cw.observe(source, extension_id="quality_snapshot")
    plan = cw.compile([source], extensions=[observation])
    path = tmp_path / "quality.jsonl"

    plan.create_session(extension_bindings={"quality_snapshot": cw.Snapshot(path)}).run()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["degraded", "invalid"]
    payload = plan.portable_ir.to_json()
    assert str(path) not in payload
    assert "quality_snapshot" in payload


def test_extension_failure_error_identifies_observation_contract() -> None:
    """FAIL policyのhandler例外へID、slot、Node、Port、policyを付ける。"""

    source = cw.Flow([1])
    plan = cw.compile(
        [source],
        extensions=[cw.observe(source, extension_id="failing_recorder")],
    )

    with pytest.raises(cw.ExtensionExecutionError) as captured:
        plan.create_session(extension_bindings={"failing_recorder": _FailingExtension()}).run()

    message = str(captured.value)
    assert "extension_id 'failing_recorder'" in message
    assert "slot 'extension:failing_recorder'" in message
    assert "node 0 port 0" in message
    assert "contract=failure_policy:fail" in message


def test_extension_descriptor_round_trip_preserves_trigger_and_slot() -> None:
    """Extension観測契約をbinding実体なしでPortablePlanIRへround-tripする。"""

    source = cw.Flow([1])
    observation = cw.observe(
        source,
        extension_id="spectrum_snapshot",
        trigger=cw.EveryLogicalTime(Fraction(5), phase=1),
        priority=3,
    )
    plan = cw.compile([source], extensions=[observation])

    restored = cw.PortablePlanIR.from_json(plan.portable_ir.to_json())
    descriptor = restored.extensions[0]

    assert restored == plan.portable_ir
    assert descriptor.extension_id == "spectrum_snapshot"
    assert descriptor.observed_port_id == source.port_id
    assert descriptor.binding_slot == "extension:spectrum_snapshot"
    assert descriptor.trigger.kind == "every_logical_time"
    assert descriptor.trigger.period is not None
    assert descriptor.trigger.period.numerator == 5
    assert descriptor.trigger.phase is not None
    assert descriptor.trigger.phase.numerator == 1
