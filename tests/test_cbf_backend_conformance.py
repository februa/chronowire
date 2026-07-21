"""Python/Cython CBF Backend交換と混在Stageを検証する。"""

from chronowire_reference import run_cbf_conformance


def test_python_cython_and_mixed_cbf_traces_are_equivalent() -> None:
    """CBF実装言語とPython境界の有無で意味論traceを変えない。"""

    python, cython, mixed = run_cbf_conformance()

    assert cython.trace == python.trace
    assert mixed.trace == python.trace
    assert [item.value for item in python.trace] == [
        ((1.0, 2.0, 3.0, 4.0),),
        ((5.0, 6.0, 7.0, 8.0),),
    ]
    assert python.trace[0].diagnostic_codes == ("CBF_REFERENCE_DEGRADED_INPUT",)


def test_cython_cbf_abi_and_mixed_stage_boundaries_are_explicit() -> None:
    """Cython ABIと前後のPython callback境界をPlanへ記録する。"""

    python, cython, mixed = run_cbf_conformance()

    assert python.kernel_abi == "python-v1"
    assert cython.kernel_abi == "chronowire.reference.fixed_cbf_f64.v1"
    assert "cython_cbf" in cython.stage_domains
    assert mixed.stage_domains.count("python") == 2
    assert "cython_cbf" in mixed.stage_domains
