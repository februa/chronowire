"""C++ Executor移行判断benchmarkの測定契約を固定する。"""

import pytest

from benchmarks.native_executor_cpp_gate import run_benchmark


def test_cpp_gate_benchmark_records_latency_copy_and_boundary_metrics() -> None:
    """性能値に閾値を置かず、必要な測定指標と静的copy量を検証する。"""

    report = run_benchmark(
        sample_count=8,
        block_sizes=(4,),
        channels=2,
        beams=1,
        warmups=0,
        repeats=2,
    )

    assert report.schema_version == "0.1"
    assert report.repeats == 2
    case = report.cases[0]
    assert case.output_frames == 2
    assert case.end_to_end.samples == 2
    assert case.end_to_end.p99_ns >= case.end_to_end.p50_ns > 0
    assert case.source_materialization_call.p50_ns > 0
    assert case.input_time_pack_call.p50_ns > 0
    assert case.scheduler_call.p50_ns > 0
    assert case.kernel_call.p50_ns > 0
    assert case.collector_reconstruction_call.p50_ns > 0
    assert case.end_to_end_samples_per_second > 0
    assert case.native_calls_p50_ns > 0
    assert case.native_calls_share_percent > 0
    assert case.accounted_calls_p50_ns >= case.native_calls_p50_ns
    assert case.unattributed_p50_ns >= 0
    assert case.accounted_calls_share_percent > 0
    assert case.python_heap_peak_bytes > 0
    assert case.copy_accounting.payload_copy_count == 5
    assert case.copy_accounting.python_native_transitions == 4
    assert case.copy_accounting.stage_python_dispatches == 1
    assert case.copy_accounting.abi_boundary_copy_bytes == 192
    assert case.copy_accounting.abi_boundary_copy_percent == pytest.approx(100.0 / 3.0)
    assert case.copy_accounting.scheduler_return_copy_bytes == 8 * 4 * 2 * 2
    assert case.copy_accounting.kernel_return_copy_bytes == 8 * 4 * 1 * 2
    assert case.copy_accounting.total_payload_copy_bytes == 576


@pytest.mark.parametrize("block_sizes", [(), (0,), (3,)])
def test_cpp_gate_benchmark_rejects_unstable_block_grids(block_sizes: tuple[int, ...]) -> None:
    """空、非正、端数frameを生むblock gridを測定条件として受理しない。"""

    with pytest.raises(ValueError, match="block size"):
        run_benchmark(sample_count=8, block_sizes=block_sizes, warmups=0, repeats=1)
