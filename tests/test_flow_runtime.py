"""Flow、compile、runtimeのv0.1契約を検証する。"""

from collections.abc import Iterator
from fractions import Fraction

import pytest

import chronowire as cw


def _values(result: cw.OutputResult[object]) -> list[object]:
    """test assertion用にEmission値だけを取り出す。"""

    return [item.value for item in result.emissions]


class _CounterSource:
    """durationで停止されるgenerated Sourceのtest実装。"""

    is_finite = False

    def read(self, request: cw.SourceRequest, config: cw.Config) -> cw.SourceBatch[int]:
        index = request.logical_start.ticks
        interval = cw.LogicalInterval(cw.LogicalTime(index), cw.LogicalTime(index + 1))
        return cw.SourceBatch((cw.Emission(index, interval, index),))


class _RequestRecordingSource:
    """RATE Nodeが決めたSourceRequest幅を記録するgenerated Source。"""

    is_finite = False

    def __init__(self) -> None:
        self.durations: list[Fraction] = []

    def read(self, request: cw.SourceRequest, config: cw.Config) -> cw.SourceBatch[int]:
        self.durations.append(request.duration)
        start = request.logical_start.as_fraction()
        end = start + request.duration
        interval = cw.LogicalInterval(
            cw.LogicalTime(start.numerator, 1, start.denominator),
            cw.LogicalTime(end.numerator, 1, end.denominator),
        )
        return cw.SourceBatch((cw.Emission(len(self.durations), interval, 0),))


class _CountingIterable:
    """Schedulerがfinite Sourceを何件先行取得したか記録する。"""

    def __init__(self, count: int) -> None:
        self.count = count
        self.yielded = 0

    def __iter__(self) -> Iterator[int]:
        for value in range(self.count):
            self.yielded += 1
            yield value


def test_fan_out_executes_common_ancestor_once() -> None:
    """分岐数によって共通前段の副作用回数が増えないことを確認する。"""

    calls: list[int] = []

    def preprocess(value: int) -> int:
        calls.append(value)
        return value + 1

    source = cw.Flow([1, 2, 3])
    base = source.map(preprocess)
    doubled = base.map(lambda value: value * 2)
    squared = base.map(lambda value: value * value)
    result = cw.compile(
        [
            cw.output(doubled, collector=cw.Bounded(3)),
            cw.output(squared, collector=cw.Bounded(3)),
        ]
    ).run()

    assert calls == [1, 2, 3]
    assert _values(result.outputs[0]) == [4, 6, 8]
    assert _values(result.outputs[1]) == [4, 9, 16]


def test_zero_one_many_emission_contract() -> None:
    """listを暗黙展開せず、skipとemit_manyだけを特殊解釈する。"""

    source = cw.Flow([0, 1, 2])

    def expand(value: int) -> object:
        if value == 0:
            return cw.skip()
        if value == 1:
            return [10, 11]
        return cw.emit_many([20, 21])

    expanded = source.map(expand, max_items=2)
    result = cw.compile([cw.output(expanded, collector=cw.Bounded(3))]).run()

    assert _values(result.outputs[0]) == [[10, 11], 20, 21]


def test_emit_many_must_fit_declared_max_items() -> None:
    """宣言上限を超える複数Emissionを配送前に拒否する。"""

    expanded = cw.Flow([1]).map(lambda value: cw.emit_many([value, value + 1]))

    with pytest.raises(cw.KernelExecutionError, match="contract max_items=1"):
        cw.compile([expanded]).run()


def test_map_rejects_nonpositive_max_items() -> None:
    """生成件数上限はGraph構築時から正の整数に限定する。"""

    with pytest.raises(ValueError, match="max_items must be positive"):
        cw.Flow([1]).map(lambda value: value, max_items=0)


def test_frame_overlap_preserves_intervals() -> None:
    """size=3/hop=2の重複frameが元入力区間を保持することを確認する。"""

    frames = cw.Flow([0, 1, 2, 3, 4]).frame(3, hop=2)
    result = cw.compile([cw.output(frames, collector=cw.Bounded(2))]).run()
    emissions = result.outputs[0].emissions

    assert [item.value for item in emissions] == [(0, 1, 2), (2, 3, 4)]
    assert emissions[0].interval == cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(3))
    assert emissions[1].interval == cw.LogicalInterval(cw.LogicalTime(2), cw.LogicalTime(5))


def test_latest_state_uses_value_at_or_before_main_interval() -> None:
    """latest stateが未来値を参照せず、同一区間までの最新値を使う。"""

    source = cw.Flow([1, 2, 3])
    state = source.map(lambda value: value * 10)
    main = source.map(lambda value: value)
    combined = main.map(
        lambda value, *, state_value: value + state_value, state_value=state.latest()
    )
    result = cw.compile([cw.output(combined, collector=cw.Bounded(3))]).run()

    assert _values(result.outputs[0]) == [11, 22, 33]


def test_exact_merge_stall_stops_unneeded_pull_before_source_eof() -> None:
    """生成不能intervalを検出した経路がfinite Sourceを末尾まで先行取得しない。"""

    values = _CountingIterable(100)
    source = cw.Flow(values)
    framed = source.frame(2)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)

    result = cw.compile([cw.output(merged, collector=cw.Latest())]).run()

    assert values.yielded == 2
    assert result.outputs[0].emissions == ()
    assert any(item.code == "STALLED_EXACT_MERGE" for item in result.diagnostics)
    assert not result.completed


def test_stalled_merge_releases_cursors_for_independent_output() -> None:
    """停止mergeのcursorを解放し、同じSourceの独立終端はEOFまで継続する。"""

    values = _CountingIterable(5)
    source = cw.Flow(values)
    framed = source.frame(2)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)
    result = cw.compile(
        [
            cw.output(merged, collector=cw.Latest()),
            cw.output(source, collector=cw.Bounded(5)),
        ]
    ).run()

    assert values.yielded == 5
    assert _values(result.outputs[1]) == [0, 1, 2, 3, 4]
    assert not any(item.code == "SCHEDULER_DEADLOCK" for item in result.diagnostics)


def test_producer_frontier_stalls_merge_after_skip_without_read_ahead() -> None:
    """Skipで必要intervalを通過した分岐をfrontierから即座に停止する。"""

    values = _CountingIterable(100)
    source = cw.Flow(values)
    main = source.map(lambda value: value)
    skipped = source.map(lambda value: cw.skip())
    merged = main.map(lambda value, *, other: (value, other), other=skipped)

    result = cw.compile([merged]).run()

    assert values.yielded == 1
    stalled = next(item for item in result.diagnostics if item.code == "STALLED_EXACT_MERGE")
    assert stalled.details["producer_frontier"] == "1"
    assert not any(item.code == "SCHEDULER_DEADLOCK" for item in result.diagnostics)


def test_frontier_compares_next_start_not_required_interval_end() -> None:
    """長いmain intervalでもauxが開始時刻を通過した時点で停止する。"""

    values = _CountingIterable(100)
    source = cw.Flow(values)
    main = source.frame(4)
    skipped = source.map(lambda value: cw.skip())
    merged = main.map(lambda value, *, other: (value, other), other=skipped)

    result = cw.compile([merged]).run()

    assert values.yielded == 4
    stalled = next(item for item in result.diagnostics if item.code == "STALLED_EXACT_MERGE")
    assert stalled.interval == cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(4))
    assert stalled.details["producer_frontier"] == "1"


def test_plan_run_resets_runtime_state() -> None:
    """同じPlanを再実行してもcollectorとframe stateを持ち越さない。"""

    frames = cw.Flow([1, 2, 3, 4]).frame(2)
    plan = cw.compile([cw.output(frames, collector=cw.Bounded(2))])

    assert _values(plan.run().outputs[0]) == [(1, 2), (3, 4)]
    assert _values(plan.run().outputs[0]) == [(1, 2), (3, 4)]


def test_generated_source_requires_and_respects_duration() -> None:
    """無限Sourceをdurationなしで走らせず、指定論理区間で停止する。"""

    source = cw.Flow(_CounterSource())
    plan = cw.compile([cw.output(source, collector=cw.Bounded(3))])

    try:
        plan.run()
    except ValueError as error:
        assert "requires run(duration" in str(error)
    else:
        raise AssertionError("generated Source must require duration")

    result = plan.run(duration=float(Fraction(3)))
    assert _values(result.outputs[0]) == [0, 1, 2]


def test_rate_fires_on_exact_rational_boundaries_with_hold_policy() -> None:
    """RATEが浮動小数誤差なしに入力interval内の周期境界で発火することを確認する。"""

    source = cw.Flow(
        [
            cw.Emission(
                10,
                cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1, 1, 2)),
                0,
            ),
            cw.Emission(
                20,
                cw.LogicalInterval(cw.LogicalTime(1, 1, 2), cw.LogicalTime(1)),
                1,
            ),
        ]
    )
    clocked = source.rate(4)
    result = cw.compile([cw.output(clocked, collector=cw.Bounded(4))]).run()
    emissions = result.outputs[0].emissions

    assert [item.value for item in emissions] == [10, 10, 20, 20]
    assert [item.interval.start.as_fraction() for item in emissions] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
    ]
    assert all(
        item.interval.end.as_fraction() - item.interval.start.as_fraction() == Fraction(1, 4)
        for item in emissions
    )


def test_rate_controls_generated_source_request_period() -> None:
    """下流RATEの周期がgenerated Sourceのpull request幅になることを確認する。"""

    source_impl = _RequestRecordingSource()
    clocked = cw.Flow(source_impl).rate(4)
    result = cw.compile([cw.output(clocked, collector=cw.Bounded(4))]).run(duration=1.0)

    assert source_impl.durations == [Fraction(1, 4)] * 4
    assert _values(result.outputs[0]) == [1, 2, 3, 4]


def test_rate_rejects_non_positive_frequency() -> None:
    """論理周期を定義できないrate指定をGraph構築時に拒否する。"""

    for frequency in (0, -1, float("inf")):
        try:
            cw.Flow([1]).rate(frequency)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid rate {frequency!r} must fail")
