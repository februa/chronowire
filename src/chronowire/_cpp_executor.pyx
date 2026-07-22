# cython: language_level=3
# distutils: language = c++

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint8_t, uint64_t, uintptr_t
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

    cdef cppclass GraphNodeSpec:
        GraphNodeSpec() except +
        int64_t node_id
        int opcode
        vector[int64_t] input_ports
        vector[int] input_semantics
        int64_t output_port
        int64_t period_numerator
        int64_t period_denominator
        int rate_policy
        size_t frame_size
        size_t frame_hop
        bint pad_end
        bint accepts_invalid
        string kernel_abi
        string process_model
        vector[double] kernel_parameters
        vector[size_t] parameter_shape
        vector[size_t] output_shape
        uintptr_t native_create
        uintptr_t native_process
        uintptr_t native_flush
        uintptr_t native_destroy

    cdef cppclass GraphOutputSpec:
        GraphOutputSpec() except +
        int64_t port_id
        int collector_kind
        size_t collector_capacity
        int overflow_policy

    cdef cppclass GraphRuntimeResult:
        vector[RuntimeResult] outputs
        vector[vector[size_t]] value_offsets
        vector[vector[size_t]] shapes
        vector[vector[size_t]] shape_offsets
        vector[vector[size_t]] provenance_offsets
        vector[vector[int64_t]] invalid_nodes
        vector[vector[size_t]] invalid_node_offsets
        vector[vector[int64_t]] degraded_nodes
        vector[vector[size_t]] degraded_node_offsets
        vector[vector[int64_t]] native_diagnostic_nodes
        vector[vector[uint8_t]] native_diagnostic_severities
        vector[vector[string]] native_diagnostic_codes
        vector[vector[string]] native_diagnostic_messages
        vector[vector[size_t]] native_diagnostic_offsets
        vector[vector[int64_t]] metadata_source_indices
        vector[int64_t] status_counts
        uint64_t scheduler_ns
        uint64_t kernel_ns
        uint64_t output_select_ns
        uint64_t owned_input_bytes
        uint64_t output_boundary_bytes
        uint64_t executed_node_count

    cdef cppclass GraphRuntimeSession:
        GraphRuntimeSession(
            const string& schema_version,
            const vector[GraphNodeSpec]& nodes,
            const vector[GraphOutputSpec]& outputs,
            const char* source_values,
            size_t source_value_bytes,
            const char* source_starts,
            size_t source_start_bytes,
            const char* source_ends,
            size_t source_end_bytes,
            const char* source_statuses,
            size_t source_status_bytes,
            const char* source_resets,
            size_t source_reset_bytes,
            size_t source_count,
            size_t source_width,
            int64_t source_timebase_denominator,
        ) except +
        GraphRuntimeResult run(
            bint has_logical_end,
            int64_t logical_end_numerator,
            int64_t logical_end_denominator,
        ) except + nogil


cdef vector[double] _f64_vector(bytes values):
    """immutable f64 bytesをC++所有vectorへcopyする。"""

    cdef vector[double] result
    cdef size_t index
    cdef size_t count
    cdef const double* pointer
    if len(values) % sizeof(double) != 0:
        raise ValueError("native parameter byte length is not aligned to float64")
    count = len(values) // sizeof(double)
    if count == 0:
        return result
    pointer = <const double*>PyBytes_AS_STRING(values)
    for index in range(count):
        result.push_back(pointer[index])
    return result


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
        # PyBytesのpointerはborrowedだが、constructorが呼出し中に所有vectorへcopyする。
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
        # このCython instanceだけがRuntimeSessionを所有し、Python objectより先に解放する。
        del self._runtime

    def run(self):
        """C++ state machineを実行して観測境界用batchと計測値を返す。"""

        cdef RuntimeResult result
        cdef size_t index
        cdef size_t item_index
        cdef list sequences = []
        cdef list starts = []
        cdef list ends = []
        cdef list statuses = []
        cdef list provenance = []
        cdef list source_indices
        cdef list status_counts = []
        cdef bytes values

        # RuntimeSessionは入力を所有しPython C APIを呼ばないため、全state machineをnogilで動かせる。
        with nogil:
            result = self._runtime.run()

        if result.values.size() == 0:
            values = b""
        else:
            # C++ collectorが保持した値だけをPython観測境界へ一度copyする。
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
            # provenanceはPython側Diagnostic復元用のSource indexであり、値bufferを所有しない。
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


cdef class CppGraphNativeSession:
    """PortablePlanIR DAGを所有して実行するC++ runtime session。"""

    cdef GraphRuntimeSession* _runtime

    def __cinit__(
        self,
        schema_version,
        nodes,
        outputs,
        bytes source_values,
        bytes source_starts,
        bytes source_ends,
        bytes source_statuses,
        bytes source_resets,
        source_count,
        source_width,
        source_timebase_denominator,
    ):
        cdef vector[GraphNodeSpec] native_nodes
        cdef vector[GraphOutputSpec] native_outputs
        cdef GraphNodeSpec node
        cdef GraphOutputSpec output
        cdef bytes schema_bytes = schema_version.encode("utf-8")
        cdef string native_schema = schema_bytes
        cdef bytes abi_bytes
        cdef bytes model_bytes
        cdef object descriptor
        cdef object extent
        cdef object input_port
        cdef object input_semantic

        self._runtime = NULL
        for descriptor in nodes:
            node = GraphNodeSpec()
            node.node_id = descriptor[0]
            node.opcode = descriptor[1]
            for input_port in descriptor[2]:
                node.input_ports.push_back(input_port)
            for input_semantic in descriptor[3]:
                node.input_semantics.push_back(input_semantic)
            node.output_port = descriptor[4]
            node.period_numerator = descriptor[5]
            node.period_denominator = descriptor[6]
            node.rate_policy = descriptor[7]
            node.frame_size = descriptor[8]
            node.frame_hop = descriptor[9]
            node.pad_end = descriptor[10]
            node.accepts_invalid = descriptor[11]
            abi_bytes = descriptor[12].encode("utf-8")
            model_bytes = descriptor[13].encode("utf-8")
            node.kernel_abi = abi_bytes
            node.process_model = model_bytes
            node.kernel_parameters = _f64_vector(descriptor[14])
            for extent in descriptor[15]:
                node.parameter_shape.push_back(extent)
            for extent in descriptor[16]:
                node.output_shape.push_back(extent)
            node.native_create = descriptor[17]
            node.native_process = descriptor[18]
            node.native_flush = descriptor[19]
            node.native_destroy = descriptor[20]
            native_nodes.push_back(node)
        for descriptor in outputs:
            output = GraphOutputSpec()
            output.port_id = descriptor[0]
            output.collector_kind = descriptor[1]
            output.collector_capacity = descriptor[2]
            output.overflow_policy = descriptor[3]
            native_outputs.push_back(output)
        # 全pointerはborrowedだが、C++ constructorが戻る前にsession所有領域へcopyする。
        self._runtime = new GraphRuntimeSession(
            native_schema,
            native_nodes,
            native_outputs,
            PyBytes_AS_STRING(source_values),
            len(source_values),
            PyBytes_AS_STRING(source_starts),
            len(source_starts),
            PyBytes_AS_STRING(source_ends),
            len(source_ends),
            PyBytes_AS_STRING(source_statuses),
            len(source_statuses),
            PyBytes_AS_STRING(source_resets),
            len(source_resets),
            source_count,
            source_width,
            source_timebase_denominator,
        )

    def __dealloc__(self):
        del self._runtime

    def run(self, logical_end_numerator=None, logical_end_denominator=None):
        """DAGを指定論理境界まで実行し、collector別batchを返す。"""

        cdef GraphRuntimeResult result
        cdef RuntimeResult output
        cdef bint has_logical_end = logical_end_numerator is not None
        cdef int64_t end_numerator = (
            0 if logical_end_numerator is None else logical_end_numerator
        )
        cdef int64_t end_denominator = (
            1 if logical_end_denominator is None else logical_end_denominator
        )
        cdef size_t output_index
        cdef size_t item_index
        cdef size_t offset
        cdef size_t start
        cdef size_t end
        cdef list python_outputs = []
        cdef list shapes
        cdef list provenance
        cdef list invalid_nodes
        cdef list degraded_nodes
        cdef list native_diagnostics
        cdef list item_native_diagnostics
        cdef bytes code_bytes
        cdef bytes message_bytes
        cdef bytes values

        # C++ sessionは全Plan/inputを所有しPython callbackを持たない。
        with nogil:
            result = self._runtime.run(has_logical_end, end_numerator, end_denominator)

        for output_index in range(result.outputs.size()):
            output = result.outputs[output_index]
            if output.values.size() == 0:
                values = b""
            else:
                values = PyBytes_FromStringAndSize(
                    <const char*>&output.values[0],
                    output.values.size() * sizeof(double),
                )
            shapes = []
            provenance = []
            invalid_nodes = []
            degraded_nodes = []
            native_diagnostics = []
            for item_index in range(output.retained_count):
                start = result.shape_offsets[output_index][item_index]
                end = result.shape_offsets[output_index][item_index + 1]
                shapes.append(
                    tuple(
                        result.shapes[output_index][offset]
                        for offset in range(start, end)
                    )
                )
                start = result.provenance_offsets[output_index][item_index]
                end = result.provenance_offsets[output_index][item_index + 1]
                provenance.append(
                    tuple(output.provenance[offset] for offset in range(start, end))
                )
                start = result.invalid_node_offsets[output_index][item_index]
                end = result.invalid_node_offsets[output_index][item_index + 1]
                invalid_nodes.append(
                    tuple(
                        result.invalid_nodes[output_index][offset]
                        for offset in range(start, end)
                    )
                )
                start = result.degraded_node_offsets[output_index][item_index]
                end = result.degraded_node_offsets[output_index][item_index + 1]
                degraded_nodes.append(
                    tuple(
                        result.degraded_nodes[output_index][offset]
                        for offset in range(start, end)
                    )
                )
                start = result.native_diagnostic_offsets[output_index][item_index]
                end = result.native_diagnostic_offsets[output_index][item_index + 1]
                item_native_diagnostics = []
                for offset in range(start, end):
                    code_bytes = result.native_diagnostic_codes[output_index][offset]
                    message_bytes = result.native_diagnostic_messages[output_index][offset]
                    item_native_diagnostics.append(
                        (
                            result.native_diagnostic_severities[output_index][offset],
                            result.native_diagnostic_nodes[output_index][offset],
                            code_bytes.decode("utf-8"),
                            message_bytes.decode("utf-8"),
                        )
                    )
                native_diagnostics.append(tuple(item_native_diagnostics))
            python_outputs.append(
                (
                    values,
                    tuple(result.value_offsets[output_index]),
                    tuple(shapes),
                    tuple(output.sequences),
                    tuple(output.starts),
                    tuple(output.ends),
                    tuple(output.statuses),
                    tuple(provenance),
                    tuple(invalid_nodes),
                    tuple(degraded_nodes),
                    tuple(native_diagnostics),
                    tuple(result.metadata_source_indices[output_index]),
                    output.timebase_denominator,
                    output.received_count,
                    output.dropped_count,
                    output.overflowed,
                )
            )
        return (
            tuple(python_outputs),
            tuple(result.status_counts),
            (
                result.scheduler_ns,
                result.kernel_ns,
                result.output_select_ns,
                result.owned_input_bytes,
                result.output_boundary_bytes,
                2,
                0,
                result.executed_node_count,
            ),
        )
