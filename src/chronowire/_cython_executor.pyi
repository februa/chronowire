def run_f64_rate_frame(
    source_values: memoryview,
    period_numerator: int,
    period_denominator: int,
    frame_size: int,
    frame_hop: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
]: ...
