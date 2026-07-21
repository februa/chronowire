"""Native Executor着手前のsample/block CBF基準を固定する。"""

from examples.native_executor_baseline import run_block_baseline, run_sample_baseline


def test_block_cbf_matches_sample_trace_with_fewer_python_kernel_calls() -> None:
    """block化が意味論を変えずPython呼出し粒度を下げる。"""

    sample = run_sample_baseline()
    block = run_block_baseline()

    assert block.emissions == sample.emissions
    assert block.measurement.kernel_calls == 2
    assert sample.measurement.kernel_calls == 10
    assert block.measurement.scheduler_steps < sample.measurement.scheduler_steps
    assert block.measurement.kernel_total_ns >= 0
    assert sample.measurement.kernel_total_ns >= 0
    assert block.measurement.buffer_high_watermark_items <= (
        sample.measurement.buffer_high_watermark_items
    )
