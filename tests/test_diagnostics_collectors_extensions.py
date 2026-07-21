"""劣化結果、collector、Extension、同期診断を検証する。"""

import json
from pathlib import Path

import pytest

import chronowire as cw


class _LifecycleExtension:
    """例外時もfinalizeされることを観測するtest Extension。"""

    priority = 0
    trigger = cw.Always()

    def __init__(self, port_id: int) -> None:
        self._port_id = port_id
        self.initialized = False
        self.finalized = False
        self.values: list[object] = []

    def observed_ports(self) -> tuple[int, ...]:
        return (self._port_id,)

    def initialize(self, context: cw.PlanContext) -> None:
        self.initialized = True

    def on_output(self, event: cw.OutputEvent) -> None:
        self.values.append(event.emission.value)

    def on_diagnostic(self, diagnostic: cw.Diagnostic) -> None:
        pass

    def finalize(self, context: cw.PlanContext) -> None:
        self.finalized = True


class _TraceExtension:
    """一つのPortについて配送順を外部traceへ記録するExtension。"""

    priority = 0
    trigger = cw.Always()

    def __init__(self, port_id: int, trace: list[str]) -> None:
        self._port_id = port_id
        self._trace = trace

    def observed_ports(self) -> tuple[int, ...]:
        return (self._port_id,)

    def initialize(self, context: cw.PlanContext) -> None:
        pass

    def on_output(self, event: cw.OutputEvent) -> None:
        self._trace.append("extension")

    def on_diagnostic(self, diagnostic: cw.Diagnostic) -> None:
        pass

    def finalize(self, context: cw.PlanContext) -> None:
        pass


def _degraded(value: int) -> cw.Emission[int]:
    """不十分な条件でも観測可能な安全値を作る。"""

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    diagnostic = cw.Diagnostic(
        cw.Severity.WARNING,
        "INSUFFICIENT_INTEGRATION",
        "fixed fallback was used",
        interval=interval,
    )
    return cw.Emission(value, interval, 0, cw.EmissionStatus.DEGRADED, (diagnostic,))


def test_degraded_status_and_diagnostic_survive_map() -> None:
    """安全な劣化結果が通常mapで例外やOKへ変換されないことを確認する。"""

    mapped = cw.Flow([_degraded(4)]).map(lambda value: value * 2)
    result = cw.compile([cw.output(mapped, collector=cw.Latest())]).run()
    emission = result.outputs[0].emissions[0]

    assert emission.value == 8
    assert emission.status is cw.EmissionStatus.DEGRADED
    assert emission.diagnostics[0].code == "INSUFFICIENT_INTEGRATION"


def test_invalid_input_skips_kernel_and_propagates() -> None:
    """INVALIDを受理しないKernelが呼ばれず、観測情報が残ることを確認する。"""

    calls = 0
    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    invalid = cw.Emission(3, interval, 0, cw.EmissionStatus.INVALID)

    def unsafe(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    mapped = cw.Flow([invalid]).map(unsafe)
    result = cw.compile([cw.output(mapped, collector=cw.Latest())]).run()
    emission = result.outputs[0].emissions[0]

    assert calls == 0
    assert emission.status is cw.EmissionStatus.INVALID
    assert emission.diagnostics[-1].code == "INVALID_INPUT_PROPAGATED"


def test_bounded_collector_defaults_to_fail() -> None:
    """暗黙dropを避けるため、容量超過を既定で例外にする。"""

    mapped = cw.Flow([1, 2]).map(lambda value: value)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(1))])

    with pytest.raises(cw.BufferOverflowError):
        plan.run()


def test_extension_finalizes_after_collector_failure() -> None:
    """run失敗時もExtensionが外部資源を閉じられることを確認する。"""

    mapped = cw.Flow([1, 2]).map(lambda value: value)
    extension = _LifecycleExtension(mapped.port_id)
    plan = cw.compile(
        [cw.output(mapped, collector=cw.Bounded(1))],
        extensions=[extension],
    )

    with pytest.raises(cw.BufferOverflowError):
        plan.run()

    assert extension.initialized
    assert extension.finalized
    assert extension.values == [1, 2]


def test_output_delivery_order_is_extension_collector_consumer() -> None:
    """Executor間で固定する一つのEmissionの配送順を確認する。"""

    trace: list[str] = []
    source = cw.Flow([1])
    consumed = source.map(lambda value: trace.append("consumer") or value)
    result = cw.compile(
        [
            cw.output(
                source,
                collector=cw.Sink(lambda emission: trace.append("collector")),
            ),
            consumed,
        ],
        extensions=[_TraceExtension(source.port_id, trace)],
    ).run()

    assert result.completed
    assert trace == ["extension", "collector", "consumer"]


def test_bounded_drop_oldest_reports_count() -> None:
    """明示drop policyが保持値とdrop件数の両方を残す。"""

    mapped = cw.Flow([1, 2, 3]).map(lambda value: value)
    collector = cw.Bounded(2, overflow=cw.OverflowPolicy.DROP_OLDEST)
    result = cw.compile([cw.output(mapped, collector=collector)]).run().outputs[0]

    assert [item.value for item in result.emissions] == [2, 3]
    assert result.received_count == 3
    assert result.dropped_count == 1


def test_bounded_rejects_block_without_concurrent_consumer() -> None:
    """run終了までconsumerがないcollectorでdeadlockするBLOCKを拒否する。"""

    with pytest.raises(ValueError):
        cw.Bounded(1, overflow=cw.OverflowPolicy.BLOCK)


def test_compile_warns_for_possible_interval_mismatch() -> None:
    """raw itemとframeの合流を停止せずcompile warningとして報告する。"""

    source = cw.Flow([1, 2, 3, 4])
    framed = source.frame(2)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)
    plan = cw.compile([cw.output(merged, collector=cw.Latest())])

    assert [item.code for item in plan.diagnostics] == ["POSSIBLE_INTERVAL_MISMATCH"]
    run_result = plan.run()
    assert any(item.code == "UNMATCHED_INTERVAL_AT_EOF" for item in run_result.diagnostics)


def test_snapshot_records_degraded_emission(tmp_path: Path) -> None:
    """NoCollectでもExtensionが劣化値と理由を保存できることを確認する。"""

    source = cw.Flow([_degraded(4)])
    path = tmp_path / "snapshot.jsonl"
    plan = cw.compile([source], extensions=[cw.Snapshot(flow=source, path=path)])
    result = plan.run()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert result.outputs[0].emissions == ()
    assert payload["status"] == "degraded"
    assert payload["diagnostics"] == ["INSUFFICIENT_INTEGRATION"]


def test_rate_preserves_status_diagnostic_and_metadata() -> None:
    """RATEが値以外のEmission契約を発火境界で失わないことを確認する。"""

    degraded = _degraded(7)
    degraded_with_metadata = cw.Emission(
        degraded.value,
        degraded.interval,
        degraded.sequence,
        degraded.status,
        degraded.diagnostics,
        {"origin": "fallback"},
    )
    result = cw.compile(
        [cw.output(cw.Flow([degraded_with_metadata]).rate(2), collector=cw.Bounded(2))]
    ).run()

    assert [item.sequence for item in result.outputs[0].emissions] == [0, 1]
    assert all(item.status is cw.EmissionStatus.DEGRADED for item in result.outputs[0].emissions)
    assert all(
        item.diagnostics[0].code == "INSUFFICIENT_INTEGRATION"
        for item in result.outputs[0].emissions
    )
    assert all(item.metadata == {"origin": "fallback"} for item in result.outputs[0].emissions)


def test_frame_padding_has_exact_interval_status_and_diagnostic_order() -> None:
    """FRAMEのEOF paddingをExecutor同値性の基準traceとして固定する。"""

    first = _degraded(1)
    second = cw.Emission(
        2,
        cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(2)),
        1,
    )
    result = cw.compile(
        [cw.output(cw.Flow([first, second]).frame(3, pad_end=True), collector=cw.Latest())]
    ).run()
    frame = result.outputs[0].emissions[0]

    assert frame.value == (1, 2, None)
    assert frame.interval == cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(3))
    assert frame.sequence == 0
    assert frame.status is cw.EmissionStatus.DEGRADED
    assert [item.code for item in frame.diagnostics] == [
        "INSUFFICIENT_INTEGRATION",
        "FRAME_PADDED_AT_EOF",
    ]
