"""v0.2の包含、overlap、tolerance同期契約を検証する。"""

from fractions import Fraction

import chronowire as cw


def _emission(value: int, start: Fraction, end: Fraction, sequence: int) -> cw.Emission[int]:
    return cw.Emission(
        value,
        cw.LogicalInterval(
            cw.LogicalTime(start.numerator, 1, start.denominator),
            cw.LogicalTime(end.numerator, 1, end.denominator),
        ),
        sequence,
    )


def _values(result: cw.RunResult) -> list[object]:
    return [item.value for item in result.outputs[0].emissions]


def test_contains_reuses_wide_interval_for_reference_emissions() -> None:
    """包含入力は同じ広いEmissionを複数reference intervalへ再利用する。"""

    source = cw.Flow([0, 1])
    wide = source.state_source([_emission(10, Fraction(0), Fraction(2), 0)]).flow
    merged = source.map(
        lambda value, *, other: (value, other),
        other=wide.synchronize(cw.InputSemantics.CONTAINS),
    )

    result = cw.compile([cw.output(merged, collector=cw.Bounded(2))]).run()

    assert _values(result) == [(0, 10), (1, 10)]


def test_overlap_uses_positive_intersection_and_lowest_sequence() -> None:
    """正のintersectionを持つ最初のsequenceを決定的に選択する。"""

    source = cw.Flow([0, 1])
    overlapping = source.state_source([_emission(20, Fraction(1, 2), Fraction(3, 2), 0)]).flow
    merged = source.map(
        lambda value, *, other: (value, other),
        other=overlapping.synchronize(cw.InputSemantics.OVERLAPS),
    )

    result = cw.compile([cw.output(merged, collector=cw.Bounded(2))]).run()

    assert _values(result) == [(0, 20), (1, 20)]


def test_tolerance_matches_both_interval_ends_and_round_trips_ir() -> None:
    """両端差がtolerance以下の候補を選び、契約をIRへ保存する。"""

    source = cw.Flow([0, 1])
    shifted = source.state_source(
        [
            _emission(30, Fraction(1, 10), Fraction(11, 10), 0),
            _emission(31, Fraction(11, 10), Fraction(21, 10), 1),
        ]
    ).flow
    merged = source.map(
        lambda value, *, other: (value, other),
        other=shifted.synchronize(
            cw.InputSemantics.TOLERANCE,
            tolerance=Fraction(1, 5),
        ),
    )
    plan = cw.compile([cw.output(merged, collector=cw.Bounded(2))])

    result = plan.run()
    restored = cw.PortablePlanIR.from_json(plan.portable_ir.to_json())
    edge = next(item for item in restored.edges if item.semantics == "tolerance")

    assert _values(result) == [(0, 30), (1, 31)]
    assert edge.tolerance is not None
    assert (edge.tolerance.numerator, edge.tolerance.denominator) == (1, 5)
    assert edge.missing_policy == "stall"
    assert edge.adapter_buffer_id is not None
    adapter = next(item for item in restored.buffers if item.buffer_id == edge.adapter_buffer_id)
    assert adapter.kind == "sync_selection"
    assert adapter.max_items == 1


def test_flexible_sync_skip_records_each_missing_reference() -> None:
    """SKIP policyはNode全体を停止せず生成不能referenceだけを診断する。"""

    source = cw.Flow([0])
    future = source.state_source([_emission(0, Fraction(10), Fraction(11), 0)]).flow
    merged = source.map(
        lambda value, *, other: (value, other),
        other=future.synchronize(
            cw.InputSemantics.OVERLAPS,
            missing=cw.MissingInputPolicy.SKIP,
        ),
    )

    result = cw.compile([cw.output(merged, collector=cw.NoCollect())]).run()

    assert result.outputs[0].received_count == 0
    assert [item.code for item in result.diagnostics].count("SYNC_INPUT_SKIPPED") == 1
    assert result.completed
