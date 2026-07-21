# cython: language_level=3
# distutils: language = c++

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint8_t, uint64_t
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cpp_runtime.hpp" namespace "chronowire::cpp_runtime":
    cdef cppclass RuntimeResult:
        vector[double] values
        vector[int64_t] sequences
        vector[int64_t] starts
        vector[int64_t] ends
        vector[uint8_t] statuses
        vector[int64_t] provenance
        vector[int64_t] status_counts
        size_t retained_count
        size_t received_count
        size_t dropped_count
        size_t output_width
        int64_t timebase_denominator
        bint overflowed
        uint64_t scheduler_ns
        uint64_t kernel_ns
        uint64_t output_select_ns
        uint64_t owned_input_bytes
        uint64_t output_boundary_bytes

    cdef cppclass RuntimeSession:
        RuntimeSession(
            const string& schema_version,
            const vector[int]& opcodes,
            const char* source_values,
            size_t source_value_bytes,
            const char* source_starts,
            size_t source_start_bytes,
            const char* source_ends,
            size_t source_end_bytes,
            const char* source_statuses,
            size_t source_status_bytes,
            size_t source_count,
            size_t source_width,
            int64_t source_timebase_denominator,
            int64_t period_numerator,
            int64_t period_denominator,
            size_t frame_size,
            size_t frame_hop,
            const string& kernel_abi,
            const string& process_model,
            const char* kernel_parameters,
            size_t kernel_parameter_bytes,
            size_t beam_count,
            size_t weight_channel_count,
            int collector_kind,
            size_t collector_capacity,
            int overflow_policy,
            int64_t source_node_id,
            int64_t rate_node_id,
            int64_t frame_node_id,
            int64_t map_node_id,
        ) except +
        RuntimeResult run() except + nogil


cdef class CppNativeSession:
    """PortablePlanIRから構築したrun-local C++ runtime session。"""

    cdef RuntimeSession* _runtime
    cdef size_t _frame_size

    def __cinit__(
        self,
        schema_version,
        opcodes,
        bytes source_values,
        bytes source_starts,
        bytes source_ends,
        bytes source_statuses,
        source_count,
        source_width,
        source_timebase_denominator,
        period_numerator,
        period_denominator,
        frame_size,
        frame_hop,
        kernel_abi,
        process_model,
        bytes kernel_parameters,
        beam_count,
        weight_channel_count,
        collector_kind,
        collector_capacity,
        overflow_policy,
        source_node_id,
        rate_node_id,
        frame_node_id,
        map_node_id,
    ):
        cdef vector[int] native_opcodes
        cdef bytes schema_bytes = schema_version.encode("utf-8")
        cdef bytes abi_bytes = kernel_abi.encode("utf-8")
        cdef bytes model_bytes = process_model.encode("utf-8")
        cdef string native_schema = schema_bytes
        cdef string native_abi = abi_bytes
        cdef string native_model = model_bytes
        cdef object opcode

        self._runtime = NULL
        self._frame_size = frame_size
        for opcode in opcodes:
            native_opcodes.push_back(opcode)
        self._runtime = new RuntimeSession(
            native_schema,
            native_opcodes,
            PyBytes_AS_STRING(source_values),
            len(source_values),
            PyBytes_AS_STRING(source_starts),
            len(source_starts),
            PyBytes_AS_STRING(source_ends),
            len(source_ends),
            PyBytes_AS_STRING(source_statuses),
            len(source_statuses),
            source_count,
            source_width,
            source_timebase_denominator,
            period_numerator,
            period_denominator,
            frame_size,
            frame_hop,
            native_abi,
            native_model,
            PyBytes_AS_STRING(kernel_parameters),
            len(kernel_parameters),
            beam_count,
            weight_channel_count,
            collector_kind,
            collector_capacity,
            overflow_policy,
            source_node_id,
            rate_node_id,
            frame_node_id,
            map_node_id,
        )

    def __dealloc__(self):
        del self._runtime

    def run(self):
        """C++ state machineを実行して観測境界用batchと計測値を返す。"""

        cdef RuntimeResult result
        cdef Py_ssize_t index
        cdef Py_ssize_t item_index
        cdef list sequences = []
        cdef list starts = []
        cdef list ends = []
        cdef list statuses = []
        cdef list provenance = []
        cdef list source_indices
        cdef list status_counts = []
        cdef bytes values

        with nogil:
            result = self._runtime.run()

        if result.values.size() == 0:
            values = b""
        else:
            values = PyBytes_FromStringAndSize(
                <const char*>&result.values[0],
                result.values.size() * sizeof(double),
            )
        for index in range(result.retained_count):
            sequences.append(result.sequences[index])
            starts.append(result.starts[index])
            ends.append(result.ends[index])
            statuses.append(result.statuses[index])
            source_indices = []
            for item_index in range(self._frame_size):
                source_indices.append(
                    result.provenance[index * self._frame_size + item_index]
                )
            provenance.append(tuple(source_indices))
        for index in range(result.status_counts.size()):
            status_counts.append(result.status_counts[index])
        return (
            values,
            result.output_width,
            tuple(sequences),
            tuple(starts),
            tuple(ends),
            tuple(statuses),
            tuple(provenance),
            result.timebase_denominator,
            result.received_count,
            result.dropped_count,
            result.overflowed,
            tuple(status_counts),
            (
                result.scheduler_ns,
                result.kernel_ns,
                result.output_select_ns,
                result.owned_input_bytes,
                result.output_boundary_bytes,
            ),
        )
