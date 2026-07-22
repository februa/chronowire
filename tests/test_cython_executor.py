"""最小Cython ExecutorとPython基準意味論を比較する。"""

from fractions import Fraction

import pytest

import chronowire as cw


def _plan(*, hop: int = 2) -> cw.Plan:
    source = cw.Flow(cw.f64_source([1.0, 2.0, 3.0]))
    frames = source.rate(2).frame(2, hop=hop).map(cw.identity_f64())
    return cw.compile([cw.output(frames, collector=cw.Bounded(8))])


def test_cython_executor_matches_python_rate_frame_identity_trace() -> None:
    """値、interval、sequence、status、DiagnosticをPython基準と一致させる。"""

    plan = _plan()

    python = plan.run(executor=cw.PythonExecutor())
    cython = plan.run(executor=cw.CythonExecutor())

    assert cython == python
    assert [item.value for item in cython.outputs[0].emissions] == [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
    ]


def test_cython_executor_matches_overlapping_frame_trace() -> None:
    """hop付き重複FRAMEも整数tick state machineで一致する。"""

    plan = _plan(hop=1)

    assert plan.run(executor=cw.CythonExecutor()) == plan.run()


def test_cython_executor_matches_empty_source_trace() -> None:
    """空SourceではPort publishもcollector出力も発生させない。"""

    source = cw.Flow(cw.f64_source([]))
    mapped = source.rate(2).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(1))])

    result = plan.run(executor=cw.CythonExecutor())
    assert result == plan.run()
    assert result.outputs[0].emissions == ()


def test_cython_executor_matches_rational_rate_ticks() -> None:
    """有理数周期を浮動小数へ丸めず整数tickで処理する。"""

    source = cw.Flow(cw.f64_source([1.0, 2.0]))
    mapped = source.rate(Fraction(3, 2)).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    result = plan.run(executor=cw.CythonExecutor())
    assert result == plan.run()
    assert result.outputs[0].emissions[0].interval.end.as_fraction() == Fraction(4, 3)


def test_native_contract_is_recorded_in_schema_v03() -> None:
    """f64 Port schemaとidentity ABIがPortablePlanIRへ固定される。"""

    ir = _plan().portable_ir

    assert all(port.value_schema_id != "python:opaque" for port in ir.ports)
    assert ir.kernel_abis[0].native_compatible
    assert ir.kernel_abis[0].process_model == "identity_f64"
    assert ir.kernel_abis[0].workspace_size_bytes == 0
    assert "native_kernel" in [stage.execution_domain for stage in ir.stages]
    assert ir.stream_item_abis[0].status_encoding == "u8:ok=0,degraded=1,invalid=2"
    assert ir.stream_item_abis[0].diagnostic_encoding == "source_provenance_index_table"
    assert {item.port_id for item in ir.native_buffers} == {item.port_id for item in ir.ports}
    assert all(item.read_only for item in ir.native_buffers)


def test_cython_executor_preserves_degraded_status_and_diagnostic() -> None:
    """安全な劣化結果と理由をnative Stage通過後も保存する。"""

    interval0 = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    diagnostic = cw.Diagnostic(
        cw.Severity.WARNING,
        "INSUFFICIENT_INTEGRATION",
        "fixed fallback was used",
        interval=interval0,
    )
    source = cw.Flow(
        cw.f64_source(
            [
                cw.Emission(
                    1.0,
                    interval0,
                    0,
                    cw.EmissionStatus.DEGRADED,
                    (diagnostic,),
                )
            ]
        )
    )
    mapped = source.rate(2).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(1))])

    result = plan.run(executor=cw.CythonExecutor())

    assert result == plan.run(executor=cw.PythonExecutor())
    assert result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED
    assert result.outputs[0].emissions[0].diagnostics == (diagnostic, diagnostic)


def test_cython_executor_preserves_invalid_skip_diagnostic() -> None:
    """INVALID入力ではKernel skipと追加DiagnosticをPython基準に一致させる。"""

    interval0 = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    interval1 = cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(2))
    source = cw.Flow(
        cw.f64_source(
            [
                cw.Emission(1.0, interval0, 0, cw.EmissionStatus.INVALID),
                cw.Emission(2.0, interval1, 1),
            ]
        )
    )
    mapped = source.rate(1).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(1))])

    result = plan.run(executor=cw.CythonExecutor())

    assert result == plan.run(executor=cw.PythonExecutor())
    emission = result.outputs[0].emissions[0]
    assert emission.status is cw.EmissionStatus.INVALID
    assert emission.diagnostics[-1].code == "INVALID_INPUT_PROPAGATED"


def test_f64_source_rejects_mixed_scalar_and_emission_forms() -> None:
    """暗黙intervalと明示intervalを同じnative Sourceで混在させない。"""

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))

    with pytest.raises(ValueError, match="must not mix"):
        cw.f64_source([1.0, cw.Emission(2.0, interval, 1)])


def test_cython_executor_normalizes_fractional_source_intervals_to_ticks() -> None:
    """Sourceの非zero有理intervalも共通timebaseへlossless変換する。"""

    first = cw.LogicalInterval(cw.LogicalTime(1, 1, 2), cw.LogicalTime(1))
    second = cw.LogicalInterval(cw.LogicalTime(1), cw.LogicalTime(3, 1, 2))
    source = cw.Flow(
        cw.f64_source(
            [
                cw.Emission(1.0, first, 0),
                cw.Emission(2.0, second, 1),
            ]
        )
    )
    mapped = source.rate(2).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    result = plan.run(executor=cw.CythonExecutor())

    assert result == plan.run(executor=cw.PythonExecutor())
    assert result.outputs[0].emissions[0].interval.start.as_fraction() == Fraction(1, 2)


def test_cython_executor_rejects_input_overrun_until_frame_reset_is_native() -> None:
    """gap後のFRAME履歴resetを未実装のまま近似実行しない。"""

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    diagnostic = cw.Diagnostic(
        cw.Severity.WARNING,
        "INPUT_OVERRUN",
        "realtime input was dropped",
        interval=interval,
    )
    source = cw.Flow(
        cw.f64_source([cw.Emission(1.0, interval, 0, cw.EmissionStatus.DEGRADED, (diagnostic,))])
    )
    mapped = source.rate(2).frame(2).map(cw.identity_f64())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(1))])

    with pytest.raises(ValueError, match="contract=gap_reset"):
        plan.run(executor=cw.CythonExecutor())


def test_cython_executor_rejects_implicit_python_values_and_callbacks() -> None:
    """native契約不足をPython fallbackせずsession作成時に拒否する。"""

    source = cw.Flow([1.0, 2.0])
    mapped = source.rate(2).frame(2).map(lambda frame: frame)
    plan = cw.compile([mapped])

    with pytest.raises(ValueError, match="requires cw.f64_source"):
        plan.run(executor=cw.CythonExecutor())


def test_cython_executor_rejects_continuous_session_explicitly() -> None:
    """未実装ContinuousSessionをPythonへ暗黙fallbackしない。"""

    with pytest.raises(cw.SessionError, match="cython_continuous_session"):
        _plan().create_continuous_session(executor=cw.CythonExecutor())
