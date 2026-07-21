def run_f64_rate_frame(
    source_values: memoryview,
    source_starts: memoryview,
    source_ends: memoryview,
    source_statuses: memoryview,
    source_resets: memoryview,
    period_ticks: int,
    timebase_denominator: int,
    frame_size: int,
    frame_hop: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    int,
    tuple[int, int, int],
]: ...
