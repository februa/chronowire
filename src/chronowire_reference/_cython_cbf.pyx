# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdlib cimport free, malloc
from cpython.bytes cimport PyBytes_FromStringAndSize


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


def run_fixed_cbf_batch(
    const double[::1] frames,
    int frame_count,
    int sample_count,
    int channel_count,
    double[::1] weights,
    int beam_count,
):
    """複数frameを一回の呼出しで固定CBF変換する。"""

    cdef int frame_index
    cdef int sample_index
    cdef int channel_index
    cdef int beam_index
    cdef double total
    cdef double* outputs = NULL

    if frame_count < 0 or sample_count <= 0 or channel_count <= 0 or beam_count <= 0:
        raise ValueError("fixed CBF batch dimensions must be positive")
    if frames.shape[0] != frame_count * sample_count * channel_count:
        raise ValueError("fixed CBF frame batch shape is inconsistent")
    if weights.shape[0] != beam_count * channel_count:
        raise ValueError("fixed CBF weight buffer shape is inconsistent")
    if frame_count:
        outputs = <double*>malloc(
            frame_count * beam_count * sample_count * sizeof(double)
        )
        if outputs == NULL:
            raise MemoryError("fixed CBF batch output allocation failed")

    try:
        with nogil:
            for frame_index in range(frame_count):
                for beam_index in range(beam_count):
                    for sample_index in range(sample_count):
                        total = 0.0
                        for channel_index in range(channel_count):
                            total += (
                                weights[beam_index * channel_count + channel_index]
                                * frames[
                                    (
                                        (frame_index * sample_count + sample_index)
                                        * channel_count
                                    )
                                    + channel_index
                                ]
                            )
                        outputs[
                            (frame_index * beam_count + beam_index) * sample_count
                            + sample_index
                        ] = total
        return PyBytes_FromStringAndSize(
            <char*>outputs,
            frame_count * beam_count * sample_count * sizeof(double),
        )
    finally:
        free(outputs)
