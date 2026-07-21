# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.stdint cimport int64_t, uint8_t
from libc.stdlib cimport free, malloc


def run_f64_rate_frame(
    double[::1] source_values,
    int64_t[::1] source_starts,
    int64_t[::1] source_ends,
    uint8_t[::1] source_statuses,
    uint8_t[::1] source_resets,
    int64_t period_ticks,
    int64_t timebase_denominator,
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
    cdef int64_t rate_status_counts[3]
    cdef uint8_t frame_status
    cdef bint has_next_fire = False
    cdef double* rate_values = NULL
    cdef int64_t* rate_starts = NULL
    cdef int64_t* rate_ends = NULL
    cdef int64_t* rate_source_indices = NULL
    cdef uint8_t* rate_statuses = NULL
    cdef double* frame_values = NULL
    cdef int64_t* frame_starts = NULL
    cdef int64_t* frame_ends = NULL
    cdef int64_t* frame_source_indices = NULL
    cdef uint8_t* frame_statuses = NULL

    rate_status_counts[0] = 0
    rate_status_counts[1] = 0
    rate_status_counts[2] = 0
    if (
        source_starts.shape[0] != source_count
        or source_ends.shape[0] != source_count
        or source_statuses.shape[0] != source_count
        or source_resets.shape[0] != source_count
    ):
        raise ValueError("Cython native Source arrays must have equal lengths")
    if period_ticks <= 0 or timebase_denominator <= 0:
        raise ValueError("Cython RATE period must be positive")
    if frame_size <= 0 or frame_hop <= 0:
        raise ValueError("Cython FRAME size and hop must be positive")

    max_rate_count = 0
    for source_index in range(source_count):
        if source_ends[source_index] < source_starts[source_index]:
            raise ValueError("Cython Source interval end precedes start")
        if source_statuses[source_index] > 2:
            raise ValueError("Cython Source status code is outside native ABI")
        max_rate_count += (
            source_ends[source_index]
            - source_starts[source_index]
            + period_ticks
            - 1
        ) // period_ticks + 1
    if max_rate_count:
        rate_values = <double*>malloc(max_rate_count * sizeof(double))
        rate_starts = <int64_t*>malloc(max_rate_count * sizeof(int64_t))
        rate_ends = <int64_t*>malloc(max_rate_count * sizeof(int64_t))
        rate_source_indices = <int64_t*>malloc(max_rate_count * sizeof(int64_t))
        rate_statuses = <uint8_t*>malloc(max_rate_count * sizeof(uint8_t))
        if (
            rate_values == NULL
            or rate_starts == NULL
            or rate_ends == NULL
            or rate_source_indices == NULL
            or rate_statuses == NULL
        ):
            free(rate_values)
            free(rate_starts)
            free(rate_ends)
            free(rate_source_indices)
            free(rate_statuses)
            raise MemoryError("Cython RATE buffer allocation failed")

    try:
        with nogil:
            for source_index in range(source_count):
                source_start = source_starts[source_index]
                source_end = source_ends[source_index]
                if not has_next_fire or source_resets[source_index]:
                    next_fire = source_start
                    has_next_fire = True
                while next_fire < source_start:
                    next_fire += period_ticks
                while next_fire < source_end:
                    rate_values[rate_count] = source_values[source_index]
                    rate_starts[rate_count] = next_fire
                    rate_ends[rate_count] = next_fire + period_ticks
                    rate_source_indices[rate_count] = source_index
                    rate_statuses[rate_count] = source_statuses[source_index]
                    rate_status_counts[source_statuses[source_index]] += 1
                    rate_count += 1
                    next_fire += period_ticks

        if rate_count >= frame_size:
            frame_count = 1 + (rate_count - frame_size) // frame_hop
        if frame_count:
            frame_values = <double*>malloc(frame_count * frame_size * sizeof(double))
            frame_starts = <int64_t*>malloc(frame_count * sizeof(int64_t))
            frame_ends = <int64_t*>malloc(frame_count * sizeof(int64_t))
            frame_source_indices = <int64_t*>malloc(
                frame_count * frame_size * sizeof(int64_t)
            )
            frame_statuses = <uint8_t*>malloc(frame_count * sizeof(uint8_t))
            if (
                frame_values == NULL
                or frame_starts == NULL
                or frame_ends == NULL
                or frame_source_indices == NULL
                or frame_statuses == NULL
            ):
                raise MemoryError("Cython FRAME buffer allocation failed")
        with nogil:
            for frame_index in range(frame_count):
                frame_start = frame_index * frame_hop
                frame_starts[frame_index] = rate_starts[frame_start]
                frame_ends[frame_index] = rate_ends[frame_start + frame_size - 1]
                frame_status = 0
                for item_index in range(frame_size):
                    frame_values[frame_index * frame_size + item_index] = (
                        rate_values[frame_start + item_index]
                    )
                    frame_source_indices[frame_index * frame_size + item_index] = (
                        rate_source_indices[frame_start + item_index]
                    )
                    if rate_statuses[frame_start + item_index] > frame_status:
                        frame_status = rate_statuses[frame_start + item_index]
                frame_statuses[frame_index] = frame_status

        frames = tuple(
            tuple(
                frame_values[frame_index * frame_size + item_index]
                for item_index in range(frame_size)
            )
            for frame_index in range(frame_count)
        )
        starts = tuple(frame_starts[frame_index] for frame_index in range(frame_count))
        ends = tuple(frame_ends[frame_index] for frame_index in range(frame_count))
        statuses = tuple(frame_statuses[frame_index] for frame_index in range(frame_count))
        provenance = tuple(
            tuple(
                frame_source_indices[frame_index * frame_size + item_index]
                for item_index in range(frame_size)
            )
            for frame_index in range(frame_count)
        )
        return (
            frames,
            starts,
            ends,
            statuses,
            provenance,
            timebase_denominator,
            tuple(rate_status_counts[item_index] for item_index in range(3)),
        )
    finally:
        free(rate_values)
        free(rate_starts)
        free(rate_ends)
        free(rate_source_indices)
        free(rate_statuses)
        free(frame_values)
        free(frame_starts)
        free(frame_ends)
        free(frame_source_indices)
        free(frame_statuses)
