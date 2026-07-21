"""最小Cython ExecutorとPython基準意味論を比較する。"""

from fractions import Fraction

import pytest

import chronowire as cw


def _plan(*, hop: int = 2) -> cw.ExecutionPlan:
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


def test_cython_executor_rejects_implicit_python_values_and_callbacks() -> None:
    """native契約不足をPython fallbackせずsession作成時に拒否する。"""

    source = cw.Flow([1.0, 2.0])
    mapped = source.rate(2).frame(2).map(lambda frame: frame)
    plan = cw.compile([mapped])

    with pytest.raises(ValueError, match="requires cw.f64_source"):
        plan.run(executor=cw.CythonExecutor())


def test_cython_executor_rejects_continuous_session_explicitly() -> None:
    """未実装PlanSessionをPythonへ暗黙fallbackしない。"""

    with pytest.raises(cw.PlanSessionError, match="cython_continuous_session"):
        _plan().create_plan_session(executor=cw.CythonExecutor())
