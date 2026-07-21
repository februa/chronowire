# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdint cimport int64_t
from libc.stdlib cimport free, malloc


def run_f64_rate_frame(
    double[::1] source_values,
    int64_t period_numerator,
    int64_t period_denominator,
    int frame_size,
    int frame_hop,
):
    """f64 SourceへHOLD RATEとFRAMEを適用し、collector境界用batchを返す。"""

    cdef Py_ssize_t source_count = source_values.shape[0]
    cdef Py_ssize_t max_rate_count
    cdef Py_ssize_t rate_count = 0
    cdef Py_ssize_t frame_count = 0
    cdef Py_ssize_t source_index
    cdef Py_ssize_t frame_index
    cdef Py_ssize_t item_index
    cdef Py_ssize_t frame_start
    cdef int64_t source_start
    cdef int64_t source_end
    cdef int64_t next_fire = 0
    cdef double* rate_values = NULL
    cdef int64_t* rate_starts = NULL
    cdef int64_t* rate_ends = NULL
    cdef double* frame_values = NULL
    cdef int64_t* frame_starts = NULL
    cdef int64_t* frame_ends = NULL

    if period_numerator <= 0 or period_denominator <= 0:
        raise ValueError("Cython RATE period must be positive")
    if frame_size <= 0 or frame_hop <= 0:
        raise ValueError("Cython FRAME size and hop must be positive")

    max_rate_count = (
        source_count * period_denominator + period_numerator - 1
    ) // period_numerator
    if max_rate_count:
        rate_values = <double*>malloc(max_rate_count * sizeof(double))
        rate_starts = <int64_t*>malloc(max_rate_count * sizeof(int64_t))
        rate_ends = <int64_t*>malloc(max_rate_count * sizeof(int64_t))
        if rate_values == NULL or rate_starts == NULL or rate_ends == NULL:
            free(rate_values)
            free(rate_starts)
            free(rate_ends)
            raise MemoryError("Cython RATE buffer allocation failed")

    try:
        with nogil:
            for source_index in range(source_count):
                source_start = source_index * period_denominator
                source_end = source_start + period_denominator
                while next_fire < source_start:
                    next_fire += period_numerator
                while next_fire < source_end:
                    rate_values[rate_count] = source_values[source_index]
                    rate_starts[rate_count] = next_fire
                    rate_ends[rate_count] = next_fire + period_numerator
                    rate_count += 1
                    next_fire += period_numerator

        if rate_count >= frame_size:
            frame_count = 1 + (rate_count - frame_size) // frame_hop
        if frame_count:
            frame_values = <double*>malloc(frame_count * frame_size * sizeof(double))
            frame_starts = <int64_t*>malloc(frame_count * sizeof(int64_t))
            frame_ends = <int64_t*>malloc(frame_count * sizeof(int64_t))
            if frame_values == NULL or frame_starts == NULL or frame_ends == NULL:
                raise MemoryError("Cython FRAME buffer allocation failed")
        with nogil:
            for frame_index in range(frame_count):
                frame_start = frame_index * frame_hop
                frame_starts[frame_index] = rate_starts[frame_start]
                frame_ends[frame_index] = rate_ends[frame_start + frame_size - 1]
                for item_index in range(frame_size):
                    frame_values[frame_index * frame_size + item_index] = (
                        rate_values[frame_start + item_index]
                    )

        frames = tuple(
            tuple(
                frame_values[frame_index * frame_size + item_index]
                for item_index in range(frame_size)
            )
            for frame_index in range(frame_count)
        )
        starts = tuple(frame_starts[frame_index] for frame_index in range(frame_count))
        ends = tuple(frame_ends[frame_index] for frame_index in range(frame_count))
        return frames, starts, ends, period_denominator, rate_count
    finally:
        free(rate_values)
        free(rate_starts)
        free(rate_ends)
        free(frame_values)
        free(frame_starts)
        free(frame_ends)
