# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdlib cimport free, malloc


def run_fixed_cbf(
    double[::1] samples,
    int sample_count,
    int channel_count,
    double[::1] weights,
    int beam_count,
):
    """sample-major入力へ固定実数線形変換を適用する。"""

    cdef int sample_index
    cdef int channel_index
    cdef int beam_index
    cdef double total
    cdef double* outputs = NULL

    if sample_count < 0 or channel_count <= 0 or beam_count <= 0:
        raise ValueError("fixed CBF dimensions must be positive")
    if samples.shape[0] != sample_count * channel_count:
        raise ValueError("fixed CBF sample buffer shape is inconsistent")
    if weights.shape[0] != beam_count * channel_count:
        raise ValueError("fixed CBF weight buffer shape is inconsistent")
    if sample_count:
        outputs = <double*>malloc(beam_count * sample_count * sizeof(double))
        if outputs == NULL:
            raise MemoryError("fixed CBF output allocation failed")

    try:
        with nogil:
            for beam_index in range(beam_count):
                for sample_index in range(sample_count):
                    total = 0.0
                    for channel_index in range(channel_count):
                        total += (
                            weights[beam_index * channel_count + channel_index]
                            * samples[sample_index * channel_count + channel_index]
                        )
                    outputs[beam_index * sample_count + sample_index] = total
        return tuple(
            tuple(
                outputs[beam_index * sample_count + sample_index]
                for sample_index in range(sample_count)
            )
            for beam_index in range(beam_count)
        )
    finally:
        free(outputs)
