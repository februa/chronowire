def run_fixed_cbf(
    samples: memoryview,
    sample_count: int,
    channel_count: int,
    weights: memoryview,
    beam_count: int,
) -> tuple[tuple[float, ...], ...]: ...
