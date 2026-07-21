"""RATEとFRAMEのcompile時境界証明を検証する。"""

from fractions import Fraction

import pytest

import chronowire as cw


def test_rate_before_frame_has_an_exact_integral_frame_grid() -> None:
    """RATE後のframe size/hopが同じ有理格子へ固定される。"""

    framed = cw.Flow([1, 2]).rate(2).frame(2)

    plan = cw.compile([framed])
    time = next(
        item for item in plan.portable_ir.times if item.time_descriptor_id == framed.port_id
    )

    assert time.exact
    assert (time.duration.numerator, time.duration.denominator) == (1, 1)
    assert (time.period.numerator, time.period.denominator) == (1, 1)


def test_rate_after_frame_is_a_compile_violation() -> None:
    """完成済みframeをHOLDで複製または棄却し得るRATEを拒否する。"""

    unstable = cw.Flow([1, 2, 3, 4]).frame(2).rate(2)

    with pytest.raises(cw.CompileError, match="contract=rate_before_frame"):
        cw.compile([unstable])


def test_rate_after_frame_lineage_through_map_is_a_compile_violation() -> None:
    """frame直後だけでなくpreserve MAPを挟む経路も検出する。"""

    unstable = cw.Flow([1, 2]).frame(2).map(sum).rate(1)

    with pytest.raises(cw.CompileError, match="move Flow.rate.*before Flow.frame"):
        cw.compile([unstable])


def test_unknown_time_grid_requires_rate_before_frame() -> None:
    """明示時間変換後の未知格子をRATEなしでframe化しない。"""

    transformed = cw.Flow([1]).map(
        cw.callable_kernel(lambda value: value, time_transform="explicit")
    )

    with pytest.raises(cw.CompileError, match="contract=stable_rate_frame_boundary"):
        cw.compile([transformed.frame(2)])

    aligned = transformed.rate(2).frame(2)
    cw.compile([aligned])


def test_explicit_resampling_boundary_can_replace_a_completed_frame_grid() -> None:
    """外部resampling Kernel後にRATEで新格子を宣言すれば再frame化できる。"""

    frames = cw.Flow([1, 2]).frame(2)
    resampled = frames.map(
        cw.callable_kernel(lambda values: sum(values), time_transform="explicit")
    )

    rebuilt = resampled.rate(4).frame(4)
    cw.compile([rebuilt])


def test_rate_sensitive_exact_merge_requires_identical_frame_grid() -> None:
    """rate由来の端数をframeで解消しない同期合流をcompile時に拒否する。"""

    source = cw.Flow([1, 2])
    half_second = source.rate(2)
    unstable = source.map(lambda value, *, other: value, other=half_second)

    with pytest.raises(cw.CompileError, match="identical duration and period"):
        cw.compile([unstable])

    aligned = source.map(
        lambda value, *, other: value,
        other=half_second.frame(2),
    )
    cw.compile([aligned])


def test_fractional_rate_ratio_must_be_absorbed_by_explicit_frames() -> None:
    """5/2倍の端数格子をsize 5対size 2のframe境界で合流可能にする。"""

    source = cw.Flow([1, 2])
    converted = source.rate(Fraction(5, 2))
    aligned = source.frame(2).map(
        lambda values, *, other: (values, other),
        other=converted.frame(5),
    )

    result = cw.compile([cw.output(aligned, collector=cw.Latest())]).run()

    assert result.completed
    assert result.outputs[0].emissions[0].interval == cw.LogicalInterval(
        cw.LogicalTime(0), cw.LogicalTime(2)
    )
