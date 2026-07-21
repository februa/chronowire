#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chronowire::cpp_runtime {

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

class RuntimeSession {
public:
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
    std::int64_t source_node_id_;
    std::int64_t rate_node_id_;
    std::int64_t frame_node_id_;
    std::int64_t map_node_id_;
};

}  // namespace chronowire::cpp_runtime

