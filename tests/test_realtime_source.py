"""REALTIME_PUSH Source、bounded ingress、GapMarker伝播を検証する。"""

from dataclasses import dataclass

import pytest

import chronowire as cw


@dataclass
class _Session:
    stopped: bool = False

    def stop(self) -> None:
        """Executor終了時の停止要求を記録する。"""

        self.stopped = True


class _BurstRealtimeSource:
    def __init__(
        self,
        count: int,
        *,
        max_items: int,
        overflow_policy: cw.RealtimeOverflowPolicy = cw.RealtimeOverflowPolicy.DROP_OLDEST,
    ) -> None:
        self.count = count
        self._max_items = max_items
        self._overflow_policy = overflow_policy
        self.sessions: list[_Session] = []

    @property
    def max_items(self) -> int:
        """ingress上限を返す。"""

        return self._max_items

    @property
    def overflow_policy(self) -> cw.RealtimeOverflowPolicy:
        """overflow規則を返す。"""

        return self._overflow_policy

    def start(
        self,
        receiver: cw.RealtimeReceiver[int],
        config: cw.Config,
    ) -> _Session:
        """決定的なburstをpushして受付を閉じる。"""

        del config
        session = _Session()
        self.sessions.append(session)
        for index in range(self.count):
            receiver.publish(
                cw.Emission(
                    index,
                    cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1)),
                    index,
                )
            )
        receiver.close()
        return session


def test_realtime_drop_oldest_degrades_next_retained_emission() -> None:
    """overflow欠落をDiagnosticと次のDEGRADED Emissionへ伝播する。"""

    source_impl = _BurstRealtimeSource(4, max_items=2)
    flow = cw.Flow(source_impl)
    plan = cw.compile([cw.output(flow, collector=cw.Bounded(2))])

    source_descriptor = plan.portable_ir.sources[0]
    ingress = next(item for item in plan.portable_ir.buffers if item.kind == "realtime_ingress")
    assert source_descriptor.mode == "realtime_push"
    assert source_descriptor.ingress_buffer_id == ingress.buffer_id
    assert source_descriptor.burst_max_items == 2
    assert source_descriptor.overflow_policy == "drop_oldest"
    assert ingress.max_items == 2

    result = plan.run()
    emissions = result.outputs[0].emissions
    overrun = next(item for item in result.diagnostics if item.code == "INPUT_OVERRUN")

    assert [item.value for item in emissions] == [2, 3]
    assert emissions[0].status is cw.EmissionStatus.DEGRADED
    assert emissions[0].metadata["input_overrun_dropped_count"] == 2
    assert emissions[1].status is cw.EmissionStatus.OK
    assert overrun.interval == cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(2))
    assert overrun.details["dropped_count"] == 2
    assert source_impl.sessions[0].stopped


def test_realtime_source_state_is_run_local() -> None:
    """同じPlanの再実行でingress、drop summary、Source sessionを共有しない。"""

    source_impl = _BurstRealtimeSource(3, max_items=1)
    plan = cw.compile([cw.output(cw.Flow(source_impl), collector=cw.Latest())])

    first = plan.run()
    second = plan.run()

    assert first.outputs == second.outputs
    assert len(source_impl.sessions) == 2
    assert all(session.stopped for session in source_impl.sessions)
    assert [item.details["total_dropped_count"] for item in first.diagnostics] == [2]
    assert [item.details["total_dropped_count"] for item in second.diagnostics] == [2]


def test_realtime_source_requires_positive_ingress_capacity() -> None:
    """boundedでないRealtime Sourceをcompile時に拒否する。"""

    flow = cw.Flow(_BurstRealtimeSource(1, max_items=0))

    with pytest.raises(cw.CompileError, match="positive max_items"):
        cw.compile([flow])


def _gap_diagnostic(start: int, end: int) -> cw.Diagnostic:
    return cw.Diagnostic(
        cw.Severity.WARNING,
        "INPUT_OVERRUN",
        "test gap",
        interval=cw.LogicalInterval(cw.LogicalTime(start), cw.LogicalTime(end)),
        details={"dropped_count": end - start},
    )


def test_frame_discards_pre_gap_history() -> None:
    """FRAMEは欠落前の未完成履歴を欠落後の値へ接続しない。"""

    values = [
        cw.Emission(
            0,
            cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
            0,
        ),
        cw.Emission(
            2,
            cw.LogicalInterval(cw.LogicalTime(2), cw.LogicalTime(3)),
            2,
            cw.EmissionStatus.DEGRADED,
            (_gap_diagnostic(1, 2),),
        ),
        cw.Emission(
            3,
            cw.LogicalInterval(cw.LogicalTime(3), cw.LogicalTime(4)),
            3,
        ),
    ]
    framed = cw.Flow(values).frame(2)

    result = cw.compile([cw.output(framed, collector=cw.Bounded(1))]).run()

    assert [item.value for item in result.outputs[0].emissions] == [(2, 3)]
    assert result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED


def test_rate_reestablishes_phase_from_first_post_gap_interval() -> None:
    """RATEは欠落前の次回発火位相を持ち越さない。"""

    values = [
        cw.Emission(
            0,
            cw.LogicalInterval(cw.LogicalTime(1, 1, 4), cw.LogicalTime(3, 1, 4)),
            0,
        ),
        cw.Emission(
            2,
            cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(3, 1, 2)),
            2,
            cw.EmissionStatus.DEGRADED,
            (_gap_diagnostic(1, 1),),
        ),
    ]
    clocked = cw.Flow(values).rate(2)

    result = cw.compile([cw.output(clocked, collector=cw.Bounded(2))]).run()

    assert [item.interval.start.as_fraction() for item in result.outputs[0].emissions] == [
        cw.LogicalTime(1, 1, 4).as_fraction(),
        cw.LogicalTime(1).as_fraction(),
    ]


class _CountingSession:
    def __init__(self) -> None:
        self.count = 0

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> int:
        """session内の呼出し回数を返す。"""

        del inputs, context
        self.count += 1
        return self.count


class _CountingCompiled:
    def create_session(self) -> _CountingSession:
        """空のrun-local counterを生成する。"""

        return _CountingSession()


class _CountingKernel:
    def compile(self, context: cw.CompileContext) -> _CountingCompiled:
        """設定に依存しないcounter factoryを返す。"""

        del context
        return _CountingCompiled()


def test_stateful_kernel_session_resets_after_gap() -> None:
    """MAPは欠落境界後にCompiledKernelSessionを作り直す。"""

    values = [
        cw.Emission(
            0,
            cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
            0,
        ),
        cw.Emission(
            2,
            cw.LogicalInterval(cw.LogicalTime(2), cw.LogicalTime(3)),
            2,
            cw.EmissionStatus.DEGRADED,
            (_gap_diagnostic(1, 2),),
        ),
    ]
    counted = cw.Flow(values).map(_CountingKernel())

    result = cw.compile([cw.output(counted, collector=cw.Bounded(2))]).run()

    assert [item.value for item in result.outputs[0].emissions] == [1, 1]
