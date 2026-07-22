"""v0.2 session profilerが実行意味論を変えないことを検証する。"""

import chronowire as cw


def test_profiler_reports_bounded_resources_without_changing_trace() -> None:
    """Profiler on/offで値、時間、status、Diagnosticを同一に保つ。"""

    mapped = cw.Flow([1, 2, 3]).map(lambda value: value * 2)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(3))])

    plain = plan.run()
    profiled = plan.run(options=cw.RuntimeOptions(profiler_enabled=True))

    assert plain.outputs == profiled.outputs
    assert plain.diagnostics == profiled.diagnostics
    assert plain.status_counts == profiled.status_counts
    assert plain.profile is None
    assert profiled.profile is not None
    assert profiled.profile.scheduler_steps > 0
    assert profiled.profile.kernels[0].call_count == 3
    assert profiled.profile.kernels[0].total_ns >= profiled.profile.kernels[0].max_ns
    assert all(
        item.current_items <= item.capacity and item.high_watermark <= item.capacity
        for item in profiled.profile.buffers
    )
    assert profiled.profile.sources[0].emitted_count == 3


def test_profiler_state_is_continuous_session_local() -> None:
    """同じPlanの別sessionへscheduler/kernel計数を持ち越さない。"""

    plan = cw.compile([cw.output(cw.Flow([1, 2]).map(lambda value: value), collector=cw.Latest())])
    options = cw.RuntimeOptions(profiler_enabled=True)

    first = plan.run(options=options)
    second = plan.run(options=options)

    assert first.profile is not None and second.profile is not None
    assert first.profile.kernels[0].call_count == 2
    assert second.profile.kernels[0].call_count == 2
    assert first.profile.scheduler_steps == second.profile.scheduler_steps
