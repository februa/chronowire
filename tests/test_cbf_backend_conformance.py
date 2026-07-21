"""Python/Cython CBF Backend交換と混在Stageを検証する。"""

import pytest

import chronowire as cw
from chronowire_reference import CythonCbfBackend, FixedCbfKernel, run_cbf_conformance


def test_python_cython_and_mixed_cbf_traces_are_equivalent() -> None:
    """CBF実装言語とPython境界の有無で意味論traceを変えない。"""

    python, cython, mixed, cython_executor = run_cbf_conformance()

    assert cython.trace == python.trace
    assert mixed.trace == python.trace
    assert cython_executor.trace == python.trace
    assert [item.value for item in python.trace] == [
        ((1.0, 2.0, 3.0, 4.0),),
        ((5.0, 6.0, 7.0, 8.0),),
    ]
    assert python.trace[0].diagnostic_codes == ("CBF_REFERENCE_DEGRADED_INPUT",)


def test_cython_cbf_abi_and_mixed_stage_boundaries_are_explicit() -> None:
    """Cython ABIと前後のPython callback境界をPlanへ記録する。"""

    python, cython, mixed, cython_executor = run_cbf_conformance()

    assert python.kernel_abi == "python-v1"
    assert cython.kernel_abi == "chronowire.reference.fixed_cbf_f64.v1"
    assert "cython_cbf" in cython.stage_domains
    assert mixed.stage_domains.count("python") == 2
    assert "cython_cbf" in mixed.stage_domains
    assert cython_executor.kernel_abi == "chronowire.reference.fixed_cbf_f64.v1"
    assert cython_executor.stage_domains == cython.stage_domains
    assert cython_executor.native_buffer_count == 4
    assert cython_executor.opaque_port_count == 0


def test_cython_executor_cbf_accepts_empty_fixed_shape_source() -> None:
    """空Sourceでもshape契約を失わず空RunResultを返す。"""

    source = cw.Flow(cw.f64_vector_source([], width=2))
    beams = source.rate(4).frame(4).map(FixedCbfKernel(((0.5, 0.5),)))
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(1))],
        backend=CythonCbfBackend(),
    )

    assert plan.run(executor=cw.CythonExecutor()) == plan.run(executor=cw.PythonExecutor())


def test_cython_executor_cbf_rejects_invalid_partition_explicitly() -> None:
    """INVALID frameをnative Kernelへ誤って渡さず未実装partitionを明示する。"""

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    source = cw.Flow(
        cw.f64_vector_source(
            [cw.Emission((1.0, 1.0), interval, 0, cw.EmissionStatus.INVALID)],
            width=2,
        )
    )
    beams = source.rate(4).frame(4).map(FixedCbfKernel(((0.5, 0.5),)))
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(1))],
        backend=CythonCbfBackend(),
    )

    with pytest.raises(ValueError, match="contract=batch_invalid_partition"):
        plan.run(executor=cw.CythonExecutor())
