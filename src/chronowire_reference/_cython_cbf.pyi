def run_fixed_cbf(
    samples: memoryview[float],
    sample_count: int,
    channel_count: int,
    weights: memoryview[float],
    beam_count: int,
) -> tuple[tuple[float, ...], ...]: ...
def run_fixed_cbf_batch(
    frames: memoryview[float],
    frame_count: int,
    sample_count: int,
    channel_count: int,
    weights: memoryview[float],
    beam_count: int,
) -> bytes: ...
