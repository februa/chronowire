#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronowire::cpp_runtime {

/**
 * @brief 一回のC++実行で得たcollector境界と意味論集計を所有する。
 *
 * 時刻列は`timebase_denominator`を分母とする論理tick、計測値はnanosecond、
 * byte集計は8-bit byte単位である。全vectorは戻り値自身が所有する。
 */
struct RuntimeResult {
    std::vector<double> values;
    std::vector<std::int64_t> sequences;
    std::vector<std::int64_t> starts;
    std::vector<std::int64_t> ends;
    std::vector<std::uint8_t> statuses;
    std::vector<std::int64_t> provenance;
    std::vector<std::int64_t> status_counts;
    std::size_t retained_count = 0;
    std::size_t received_count = 0;
    std::size_t dropped_count = 0;
    std::size_t output_width = 0;
    std::int64_t timebase_denominator = 1;
    bool overflowed = false;
    std::uint64_t scheduler_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t output_select_ns = 0;
    std::uint64_t owned_input_bytes = 0;
    std::uint64_t output_boundary_bytes = 0;
};

/**
 * @brief compile済み線形Planのimmutable入力とKernel定数を所有するrun-local session。
 *
 * constructorは入力pointer範囲を検証してsession内vectorへcopyするため、callerはconstructor
 * 完了後に元bufferを解放できる。各`run()`はcursorとcollector状態をlocalに生成し、前回実行の
 * mutable状態を持ち越さない。契約違反と整数overflowは標準C++例外で報告する。
 */
class RuntimeSession {
public:
    /**
     * @brief PortablePlanIR descriptorとprocess-local bindingからsessionを構築する。
     *
     * `source_*`と`kernel_parameters`のpointerは対応するbyte長の範囲で呼出し中だけ有効で
     * なければならない。source tickは`source_timebase_denominator`を分母とし、RATE periodは
     * numerator/denominatorの論理秒で表す。
     */
    RuntimeSession(
        const std::string& schema_version,
        const std::vector<int>& opcodes,
        const char* source_values,
        std::size_t source_value_bytes,
        const char* source_starts,
        std::size_t source_start_bytes,
        const char* source_ends,
        std::size_t source_end_bytes,
        const char* source_statuses,
        std::size_t source_status_bytes,
        std::size_t source_count,
        std::size_t source_width,
        std::int64_t source_timebase_denominator,
        std::int64_t period_numerator,
        std::int64_t period_denominator,
        std::size_t frame_size,
        std::size_t frame_hop,
        const std::string& kernel_abi,
        const std::string& process_model,
        const char* kernel_parameters,
        std::size_t kernel_parameter_bytes,
        std::size_t beam_count,
        std::size_t weight_channel_count,
        int collector_kind,
        std::size_t collector_capacity,
        int overflow_policy,
        std::int64_t source_node_id,
        std::int64_t rate_node_id,
        std::int64_t frame_node_id,
        std::int64_t map_node_id
    );

    /**
     * @brief RATE、FRAME、固定CBF、collector保持選択をPythonへcallbackせず実行する。
     * @return 呼出しごとに独立した所有結果。
     * @throws std::invalid_argument Planまたはruntime契約が不正な場合。
     * @throws std::overflow_error 論理時刻またはsize計算が表現範囲を超える場合。
     */
    RuntimeResult run() const;

private:
    std::vector<double> source_values_;
    std::vector<std::int64_t> source_starts_;
    std::vector<std::int64_t> source_ends_;
    std::vector<std::uint8_t> source_statuses_;
    std::vector<double> weights_;
    std::size_t source_count_;
    std::size_t source_width_;
    std::int64_t source_timebase_denominator_;
    std::int64_t period_numerator_;
    std::int64_t period_denominator_;
    std::size_t frame_size_;
    std::size_t frame_hop_;
    std::size_t beam_count_;
    int collector_kind_;
    std::size_t collector_capacity_;
    int overflow_policy_;
};

/** @brief PortablePlanIRの一NodeをC++ DAG runtimeへ渡す固定descriptor。 */
struct GraphNodeSpec {
    std::int64_t node_id = -1;
    int opcode = -1;
    std::vector<std::int64_t> input_ports;
    std::vector<int> input_semantics;
    std::int64_t output_port = -1;
    std::int64_t period_numerator = 0;
    std::int64_t period_denominator = 1;
    int rate_policy = 0;
    std::size_t frame_size = 0;
    std::size_t frame_hop = 0;
    bool pad_end = false;
    bool accepts_invalid = false;
    std::string kernel_abi;
    std::string process_model;
    std::vector<double> kernel_parameters;
    std::vector<std::size_t> parameter_shape;
    std::vector<std::size_t> output_shape;
};

/** @brief 一つの観測Portに適用するnative collector descriptor。 */
struct GraphOutputSpec {
    std::int64_t port_id = -1;
    int collector_kind = -1;
    std::size_t collector_capacity = 0;
    int overflow_policy = 0;
};

/**
 * @brief DAG runtime全体の結果。outputsはGraphOutputSpecと同じ順序で所有する。
 *
 * 可変shape INVALID pass-throughのため、各RuntimeResultのvaluesは`value_offsets`と
 * `shape_offsets`でitem境界を表す。provenanceとinvalid_nodesも同様にoffset列を持つ。
 */
struct GraphRuntimeResult {
    std::vector<RuntimeResult> outputs;
    std::vector<std::vector<std::size_t>> value_offsets;
    std::vector<std::vector<std::size_t>> shapes;
    std::vector<std::vector<std::size_t>> shape_offsets;
    std::vector<std::vector<std::size_t>> provenance_offsets;
    std::vector<std::vector<std::int64_t>> invalid_nodes;
    std::vector<std::vector<std::size_t>> invalid_node_offsets;
    std::vector<std::vector<std::int64_t>> degraded_nodes;
    std::vector<std::vector<std::size_t>> degraded_node_offsets;
    std::vector<std::vector<std::int64_t>> metadata_source_indices;
    std::vector<std::int64_t> status_counts;
    std::uint64_t scheduler_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t output_select_ns = 0;
    std::uint64_t owned_input_bytes = 0;
    std::uint64_t output_boundary_bytes = 0;
    std::uint64_t executed_node_count = 0;
};

/**
 * @brief PortablePlanIRのSOURCE/RATE/FRAME/MAP DAGを自立運用するrun-local session。
 *
 * Source buffer、reset境界、Kernel parameter、Node/Output descriptorをconstructor中に
 * 所有領域へcopyする。各Portは一度だけ評価し、fan-out consumerは同じimmutable batchを読む。
 */
class GraphRuntimeSession {
public:
    GraphRuntimeSession(
        const std::string& schema_version,
        const std::vector<GraphNodeSpec>& nodes,
        const std::vector<GraphOutputSpec>& outputs,
        const char* source_values,
        std::size_t source_value_bytes,
        const char* source_starts,
        std::size_t source_start_bytes,
        const char* source_ends,
        std::size_t source_end_bytes,
        const char* source_statuses,
        std::size_t source_status_bytes,
        const char* source_resets,
        std::size_t source_reset_bytes,
        std::size_t source_count,
        std::size_t source_width,
        std::int64_t source_timebase_denominator
    );

    /** @brief 全有限入力、または排他的logical_end以前の入力をDAGへ流す。 */
    GraphRuntimeResult run(
        bool has_logical_end,
        std::int64_t logical_end_numerator,
        std::int64_t logical_end_denominator
    ) const;

private:
    std::vector<GraphNodeSpec> nodes_;
    std::vector<GraphOutputSpec> outputs_;
    std::vector<double> source_values_;
    std::vector<std::int64_t> source_starts_;
    std::vector<std::int64_t> source_ends_;
    std::vector<std::uint8_t> source_statuses_;
    std::vector<std::uint8_t> source_resets_;
    std::size_t source_count_;
    std::size_t source_width_;
    std::int64_t source_timebase_denominator_;
};

}  // namespace chronowire::cpp_runtime
