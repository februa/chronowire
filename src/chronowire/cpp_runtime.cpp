#include "cpp_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace chronowire::cpp_runtime {
namespace {

std::size_t checked_size_multiply(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error("CppExecutor contract=native_size_product");
    }
    return left * right;
}

std::size_t checked_size_add(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error("CppExecutor contract=native_size_sum");
    }
    return left + right;
}

template <typename T>
std::vector<T> copy_buffer(const char* data, std::size_t byte_count, std::size_t count) {
    if (byte_count != checked_size_multiply(count, sizeof(T))) {
        throw std::invalid_argument("CppExecutor contract=binding_byte_length");
    }
    std::vector<T> result(count);
    if (byte_count != 0) {
        if (data == nullptr) {
            throw std::invalid_argument("CppExecutor contract=binding_pointer");
        }
        std::memcpy(result.data(), data, byte_count);
    }
    return result;
}

std::int64_t checked_lcm(std::int64_t left, std::int64_t right) {
    if (left <= 0 || right <= 0) {
        throw std::invalid_argument("CppExecutor contract=positive_timebase");
    }
    const std::int64_t divisor = std::gcd(left, right);
    const __int128 value = static_cast<__int128>(left / divisor) * right;
    if (value > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("CppExecutor contract=signed_i64_timebase");
    }
    return static_cast<std::int64_t>(value);
}

std::int64_t checked_scale(std::int64_t value, std::int64_t factor) {
    const __int128 scaled = static_cast<__int128>(value) * factor;
    if (scaled < std::numeric_limits<std::int64_t>::min() ||
        scaled > std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("CppExecutor contract=signed_i64_ticks");
    }
    return static_cast<std::int64_t>(scaled);
}

std::uint64_t elapsed_ns(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end
) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
    );
}

}  // namespace

RuntimeSession::RuntimeSession(
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
) :
    source_values_(copy_buffer<double>(
        source_values,
        source_value_bytes,
        checked_size_multiply(source_count, source_width)
    )),
    source_starts_(copy_buffer<std::int64_t>(
        source_starts,
        source_start_bytes,
        source_count
    )),
    source_ends_(copy_buffer<std::int64_t>(
        source_ends,
        source_end_bytes,
        source_count
    )),
    source_statuses_(copy_buffer<std::uint8_t>(
        source_statuses,
        source_status_bytes,
        source_count
    )),
    weights_(copy_buffer<double>(
        kernel_parameters,
        kernel_parameter_bytes,
        checked_size_multiply(beam_count, weight_channel_count)
    )),
    source_count_(source_count),
    source_width_(source_width),
    source_timebase_denominator_(source_timebase_denominator),
    period_numerator_(period_numerator),
    period_denominator_(period_denominator),
    frame_size_(frame_size),
    frame_hop_(frame_hop),
    beam_count_(beam_count),
    collector_kind_(collector_kind),
    collector_capacity_(collector_capacity),
    overflow_policy_(overflow_policy) {
    if (schema_version != "0.3") {
        throw std::invalid_argument("CppExecutor contract=portable_plan_schema");
    }
    if (opcodes != std::vector<int>({0, 1, 2, 3})) {
        throw std::invalid_argument("CppExecutor contract=linear_native_stage");
    }
    if (source_width == 0 || frame_size == 0 || frame_hop == 0 || beam_count == 0) {
        throw std::invalid_argument("CppExecutor contract=positive_fixed_shape");
    }
    if (source_timebase_denominator <= 0 || period_numerator <= 0 || period_denominator <= 0) {
        throw std::invalid_argument("CppExecutor contract=positive_logical_time");
    }
    if (weight_channel_count != source_width) {
        throw std::invalid_argument("CppExecutor contract=kernel_channel_shape");
    }
    if (kernel_abi != "chronowire.reference.fixed_cbf_f64.v1" ||
        process_model != "fixed_cbf_f64_frame") {
        throw std::invalid_argument("CppExecutor contract=kernel_abi");
    }
    if (collector_kind < 0 || collector_kind > 2) {
        throw std::invalid_argument("CppExecutor contract=collector_kind");
    }
    if (collector_kind == 2 && collector_capacity == 0) {
        throw std::invalid_argument("CppExecutor contract=collector_capacity");
    }
    if (overflow_policy < 0 || overflow_policy > 2) {
        throw std::invalid_argument("CppExecutor contract=collector_overflow_policy");
    }
    if (source_node_id < 0 || rate_node_id < 0 || frame_node_id < 0 || map_node_id < 0) {
        throw std::invalid_argument("CppExecutor contract=nonnegative_node_id");
    }
    for (std::size_t index = 0; index < source_count_; ++index) {
        if (source_ends_[index] < source_starts_[index]) {
            throw std::invalid_argument("CppExecutor contract=source_interval_order");
        }
        if (source_statuses_[index] > 1) {
            throw std::invalid_argument("CppExecutor contract=batch_invalid_partition");
        }
    }
}

RuntimeResult RuntimeSession::run() const {
    RuntimeResult result;
    result.status_counts.assign(3, 0);
    result.output_width = checked_size_multiply(beam_count_, frame_size_);
    result.owned_input_bytes = 0;
    for (const std::size_t byte_count : {
             checked_size_multiply(source_values_.size(), sizeof(double)),
             checked_size_multiply(source_starts_.size(), sizeof(std::int64_t)),
             checked_size_multiply(source_ends_.size(), sizeof(std::int64_t)),
             checked_size_multiply(source_statuses_.size(), sizeof(std::uint8_t)),
             checked_size_multiply(weights_.size(), sizeof(double)),
         }) {
        result.owned_input_bytes = checked_size_add(result.owned_input_bytes, byte_count);
    }

    const auto scheduler_start = std::chrono::steady_clock::now();
    const std::int64_t timebase = checked_lcm(
        source_timebase_denominator_,
        period_denominator_
    );
    const std::int64_t source_scale = timebase / source_timebase_denominator_;
    const std::int64_t period_scale = timebase / period_denominator_;
    const std::int64_t period_ticks = checked_scale(period_numerator_, period_scale);
    if (period_ticks <= 0) {
        throw std::invalid_argument("CppExecutor contract=positive_rate_period");
    }

    std::vector<double> rate_values;
    std::vector<std::int64_t> rate_starts;
    std::vector<std::int64_t> rate_ends;
    std::vector<std::int64_t> rate_source_indices;
    std::vector<std::uint8_t> rate_statuses;
    rate_values.reserve(source_values_.size());
    rate_starts.reserve(source_count_);
    rate_ends.reserve(source_count_);
    rate_source_indices.reserve(source_count_);
    rate_statuses.reserve(source_count_);

    bool has_next_fire = false;
    std::int64_t next_fire = 0;
    for (std::size_t source_index = 0; source_index < source_count_; ++source_index) {
        const std::int64_t source_start = checked_scale(source_starts_[source_index], source_scale);
        const std::int64_t source_end = checked_scale(source_ends_[source_index], source_scale);
        result.status_counts[source_statuses_[source_index]] += 1;
        if (!has_next_fire) {
            next_fire = source_start;
            has_next_fire = true;
        }
        while (next_fire < source_start) {
            if (next_fire > std::numeric_limits<std::int64_t>::max() - period_ticks) {
                throw std::overflow_error("CppExecutor contract=rate_tick_overflow");
            }
            next_fire += period_ticks;
        }
        while (next_fire < source_end) {
            const std::size_t source_offset = source_index * source_width_;
            rate_values.insert(
                rate_values.end(),
                source_values_.begin() + static_cast<std::ptrdiff_t>(source_offset),
                source_values_.begin() + static_cast<std::ptrdiff_t>(source_offset + source_width_)
            );
            rate_starts.push_back(next_fire);
            if (next_fire > std::numeric_limits<std::int64_t>::max() - period_ticks) {
                throw std::overflow_error("CppExecutor contract=rate_interval_overflow");
            }
            rate_ends.push_back(next_fire + period_ticks);
            rate_source_indices.push_back(static_cast<std::int64_t>(source_index));
            rate_statuses.push_back(source_statuses_[source_index]);
            result.status_counts[source_statuses_[source_index]] += 1;
            next_fire += period_ticks;
        }
    }

    const std::size_t rate_count = rate_starts.size();
    const std::size_t frame_count =
        rate_count < frame_size_ ? 0 : 1 + (rate_count - frame_size_) / frame_hop_;
    const std::size_t frame_item_count = checked_size_multiply(frame_count, frame_size_);
    std::vector<double> frame_values(
        checked_size_multiply(frame_item_count, source_width_)
    );
    std::vector<std::int64_t> frame_starts(frame_count);
    std::vector<std::int64_t> frame_ends(frame_count);
    std::vector<std::uint8_t> frame_statuses(frame_count);
    std::vector<std::int64_t> frame_provenance(frame_item_count);
    for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        const std::size_t frame_start = frame_index * frame_hop_;
        frame_starts[frame_index] = rate_starts[frame_start];
        frame_ends[frame_index] = rate_ends[frame_start + frame_size_ - 1];
        std::uint8_t frame_status = 0;
        for (std::size_t item_index = 0; item_index < frame_size_; ++item_index) {
            const std::size_t rate_index = frame_start + item_index;
            const std::size_t source_offset = rate_index * source_width_;
            const std::size_t frame_offset =
                (frame_index * frame_size_ + item_index) * source_width_;
            std::copy_n(
                rate_values.begin() + static_cast<std::ptrdiff_t>(source_offset),
                source_width_,
                frame_values.begin() + static_cast<std::ptrdiff_t>(frame_offset)
            );
            frame_provenance[frame_index * frame_size_ + item_index] =
                rate_source_indices[rate_index];
            frame_status = std::max(frame_status, rate_statuses[rate_index]);
        }
        frame_statuses[frame_index] = frame_status;
        result.status_counts[frame_status] += 2;
    }
    const auto scheduler_end = std::chrono::steady_clock::now();

    const auto kernel_start = std::chrono::steady_clock::now();
    std::vector<double> all_outputs(
        checked_size_multiply(frame_count, result.output_width)
    );
    for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        for (std::size_t beam_index = 0; beam_index < beam_count_; ++beam_index) {
            for (std::size_t sample_index = 0; sample_index < frame_size_; ++sample_index) {
                double total = 0.0;
                for (std::size_t channel_index = 0; channel_index < source_width_; ++channel_index) {
                    total +=
                        weights_[beam_index * source_width_ + channel_index] *
                        frame_values[
                            (frame_index * frame_size_ + sample_index) * source_width_ +
                            channel_index
                        ];
                }
                all_outputs[
                    (frame_index * beam_count_ + beam_index) * frame_size_ + sample_index
                ] = total;
            }
        }
    }
    const auto kernel_end = std::chrono::steady_clock::now();

    const auto select_start = std::chrono::steady_clock::now();
    result.received_count = frame_count;
    std::size_t first_retained = 0;
    std::size_t retained_count = 0;
    if (collector_kind_ == 0) {
        result.dropped_count = frame_count;
    } else if (collector_kind_ == 1) {
        retained_count = frame_count == 0 ? 0 : 1;
        first_retained = frame_count == 0 ? 0 : frame_count - 1;
        result.dropped_count = frame_count - retained_count;
    } else if (frame_count <= collector_capacity_) {
        retained_count = frame_count;
    } else if (overflow_policy_ == 0) {
        result.overflowed = true;
    } else if (overflow_policy_ == 1) {
        retained_count = collector_capacity_;
        first_retained = frame_count - collector_capacity_;
        result.dropped_count = frame_count - collector_capacity_;
    } else {
        retained_count = collector_capacity_;
        result.dropped_count = frame_count - collector_capacity_;
    }

    result.retained_count = retained_count;
    if (!result.overflowed && retained_count != 0) {
        const std::size_t value_begin = checked_size_multiply(
            first_retained,
            result.output_width
        );
        const std::size_t value_end = checked_size_add(
            value_begin,
            checked_size_multiply(retained_count, result.output_width)
        );
        result.values.assign(
            all_outputs.begin() + static_cast<std::ptrdiff_t>(value_begin),
            all_outputs.begin() + static_cast<std::ptrdiff_t>(value_end)
        );
        for (std::size_t offset = 0; offset < retained_count; ++offset) {
            const std::size_t frame_index = first_retained + offset;
            result.sequences.push_back(static_cast<std::int64_t>(frame_index));
            result.starts.push_back(frame_starts[frame_index]);
            result.ends.push_back(frame_ends[frame_index]);
            result.statuses.push_back(frame_statuses[frame_index]);
            const std::size_t provenance_start = frame_index * frame_size_;
            result.provenance.insert(
                result.provenance.end(),
                frame_provenance.begin() + static_cast<std::ptrdiff_t>(provenance_start),
                frame_provenance.begin() +
                    static_cast<std::ptrdiff_t>(provenance_start + frame_size_)
            );
        }
    }
    result.timebase_denominator = timebase;
    result.output_boundary_bytes = checked_size_multiply(
        result.values.size(),
        sizeof(double)
    );
    const auto select_end = std::chrono::steady_clock::now();
    result.scheduler_ns = elapsed_ns(scheduler_start, scheduler_end);
    result.kernel_ns = elapsed_ns(kernel_start, kernel_end);
    result.output_select_ns = elapsed_ns(select_start, select_end);
    return result;
}

namespace {

struct GraphItem {
    std::vector<double> values;
    std::vector<std::size_t> shape;
    std::int64_t start = 0;
    std::int64_t end = 0;
    std::int64_t sequence = 0;
    std::uint8_t status = 0;
    std::vector<std::int64_t> provenance;
    std::vector<std::int64_t> invalid_nodes;
    std::int64_t metadata_source_index = -1;
    std::int64_t segment = 0;
};

using GraphBatch = std::vector<GraphItem>;

std::size_t checked_shape_product(const std::vector<std::size_t>& shape) {
    std::size_t result = 1;
    for (const std::size_t extent : shape) {
        if (extent == 0 || result > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("CppExecutor contract=native_shape_product");
        }
        result *= extent;
    }
    return result;
}

enum class GraphKernelKind { identity_f64, fixed_cbf_f64 };

GraphKernelKind resolve_graph_kernel(const GraphNodeSpec& node) {
    if (node.kernel_abi == "chronowire.kernel.identity_f64.v1" &&
        node.process_model == "identity_f64") {
        return GraphKernelKind::identity_f64;
    }
    if (node.kernel_abi == "chronowire.reference.fixed_cbf_f64.v1" &&
        node.process_model == "fixed_cbf_f64_frame") {
        return GraphKernelKind::fixed_cbf_f64;
    }
    throw std::invalid_argument("CppExecutor contract=kernel_abi_table");
}

GraphBatch run_rate_node(
    const GraphBatch& input,
    const GraphNodeSpec& node,
    std::int64_t timebase
) {
    const std::int64_t period_scale = timebase / node.period_denominator;
    const std::int64_t period_ticks = checked_scale(node.period_numerator, period_scale);
    if (period_ticks <= 0) {
        throw std::invalid_argument("CppExecutor contract=positive_rate_period");
    }
    GraphBatch output;
    bool has_next_fire = false;
    std::int64_t next_fire = 0;
    std::int64_t current_segment = -1;
    for (const GraphItem& item : input) {
        if (!has_next_fire || item.segment != current_segment) {
            next_fire = item.start;
            has_next_fire = true;
            current_segment = item.segment;
        }
        while (next_fire < item.start) {
            if (next_fire > std::numeric_limits<std::int64_t>::max() - period_ticks) {
                throw std::overflow_error("CppExecutor contract=rate_tick_overflow");
            }
            next_fire += period_ticks;
        }
        while (next_fire < item.end) {
            if (next_fire > std::numeric_limits<std::int64_t>::max() - period_ticks) {
                throw std::overflow_error("CppExecutor contract=rate_interval_overflow");
            }
            GraphItem emitted = item;
            emitted.start = next_fire;
            emitted.end = next_fire + period_ticks;
            emitted.sequence = static_cast<std::int64_t>(output.size());
            output.push_back(std::move(emitted));
            next_fire += period_ticks;
        }
    }
    return output;
}

GraphBatch run_frame_node(const GraphBatch& input, const GraphNodeSpec& node) {
    if (node.frame_size == 0 || node.frame_hop == 0 || node.pad_end) {
        throw std::invalid_argument("CppExecutor contract=fixed_unpadded_frame");
    }
    GraphBatch output;
    std::size_t segment_start = 0;
    while (segment_start < input.size()) {
        std::size_t segment_end = segment_start + 1;
        while (segment_end < input.size() &&
               input[segment_end].segment == input[segment_start].segment) {
            ++segment_end;
        }
        std::size_t offset = segment_start;
        while (node.frame_size <= segment_end - offset) {
            GraphItem frame;
            frame.start = input[offset].start;
            frame.end = input[offset + node.frame_size - 1].end;
            frame.sequence = static_cast<std::int64_t>(output.size());
            frame.segment = input[offset].segment;
            frame.shape.push_back(node.frame_size);
            frame.shape.insert(
                frame.shape.end(),
                input[offset].shape.begin(),
                input[offset].shape.end()
            );
            for (std::size_t item_index = 0; item_index < node.frame_size; ++item_index) {
                const GraphItem& item = input[offset + item_index];
                frame.values.insert(frame.values.end(), item.values.begin(), item.values.end());
                frame.status = std::max(frame.status, item.status);
                frame.provenance.insert(
                    frame.provenance.end(), item.provenance.begin(), item.provenance.end()
                );
                frame.invalid_nodes.insert(
                    frame.invalid_nodes.end(), item.invalid_nodes.begin(), item.invalid_nodes.end()
                );
            }
            output.push_back(std::move(frame));
            if (node.frame_hop > segment_end - offset) {
                break;
            }
            offset += node.frame_hop;
        }
        segment_start = segment_end;
    }
    return output;
}

GraphBatch run_map_node(const GraphBatch& input, const GraphNodeSpec& node) {
    const GraphKernelKind kind = resolve_graph_kernel(node);
    GraphBatch output;
    output.reserve(input.size());
    for (const GraphItem& item : input) {
        GraphItem mapped = item;
        mapped.sequence = static_cast<std::int64_t>(output.size());
        if (item.status == 2 && !node.accepts_invalid) {
            mapped.invalid_nodes.push_back(node.node_id);
            output.push_back(std::move(mapped));
            continue;
        }
        if (kind == GraphKernelKind::identity_f64) {
            if (node.output_shape != item.shape) {
                throw std::invalid_argument("CppExecutor contract=identity_output_shape");
            }
            output.push_back(std::move(mapped));
            continue;
        }
        if (node.parameter_shape.size() != 2 || item.shape.size() != 2 ||
            node.output_shape.size() != 2) {
            throw std::invalid_argument("CppExecutor contract=fixed_cbf_shape");
        }
        const std::size_t beam_count = node.parameter_shape[0];
        const std::size_t channel_count = node.parameter_shape[1];
        const std::size_t sample_count = item.shape[0];
        if (item.shape[1] != channel_count || node.output_shape[0] != beam_count ||
            node.output_shape[1] != sample_count ||
            node.kernel_parameters.size() != checked_size_multiply(beam_count, channel_count)) {
            throw std::invalid_argument("CppExecutor contract=fixed_cbf_shape");
        }
        mapped.values.assign(checked_size_multiply(beam_count, sample_count), 0.0);
        mapped.shape = node.output_shape;
        for (std::size_t beam = 0; beam < beam_count; ++beam) {
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                double total = 0.0;
                for (std::size_t channel = 0; channel < channel_count; ++channel) {
                    total += node.kernel_parameters[beam * channel_count + channel] *
                        item.values[sample * channel_count + channel];
                }
                mapped.values[beam * sample_count + sample] = total;
            }
        }
        output.push_back(std::move(mapped));
    }
    return output;
}

void append_offsets(
    const std::vector<std::int64_t>& values,
    std::vector<std::int64_t>& destination,
    std::vector<std::size_t>& offsets
) {
    destination.insert(destination.end(), values.begin(), values.end());
    offsets.push_back(destination.size());
}

}  // namespace

GraphRuntimeSession::GraphRuntimeSession(
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
) :
    nodes_(nodes),
    outputs_(outputs),
    source_values_(copy_buffer<double>(
        source_values,
        source_value_bytes,
        checked_size_multiply(source_count, source_width)
    )),
    source_starts_(copy_buffer<std::int64_t>(source_starts, source_start_bytes, source_count)),
    source_ends_(copy_buffer<std::int64_t>(source_ends, source_end_bytes, source_count)),
    source_statuses_(copy_buffer<std::uint8_t>(
        source_statuses, source_status_bytes, source_count
    )),
    source_resets_(copy_buffer<std::uint8_t>(source_resets, source_reset_bytes, source_count)),
    source_count_(source_count),
    source_width_(source_width),
    source_timebase_denominator_(source_timebase_denominator) {
    if (schema_version != "0.3" || nodes_.empty() || outputs_.empty()) {
        throw std::invalid_argument("CppExecutor contract=portable_graph_plan");
    }
    if (source_width_ == 0 || source_timebase_denominator_ <= 0) {
        throw std::invalid_argument("CppExecutor contract=positive_source_shape_timebase");
    }
    std::size_t source_nodes = 0;
    std::unordered_map<std::int64_t, bool> produced_ports;
    for (const GraphNodeSpec& node : nodes_) {
        if (node.node_id < 0 || node.output_port < 0 || node.opcode < 0 || node.opcode > 3) {
            throw std::invalid_argument("CppExecutor contract=portable_graph_node");
        }
        if (produced_ports.find(node.output_port) != produced_ports.end()) {
            throw std::invalid_argument("CppExecutor contract=unique_native_port");
        }
        if (node.opcode == 0) {
            ++source_nodes;
        } else if (produced_ports.find(node.input_port) == produced_ports.end()) {
            throw std::invalid_argument("CppExecutor contract=topological_native_edge");
        }
        if (node.opcode == 3) {
            static_cast<void>(resolve_graph_kernel(node));
            if (checked_shape_product(node.output_shape) == 0) {
                throw std::invalid_argument("CppExecutor contract=kernel_output_shape");
            }
        }
        produced_ports.emplace(node.output_port, true);
    }
    if (source_nodes != 1) {
        throw std::invalid_argument("CppExecutor contract=single_native_source");
    }
    for (const GraphOutputSpec& output : outputs_) {
        if (produced_ports.find(output.port_id) == produced_ports.end() ||
            output.collector_kind < 0 ||
            output.collector_kind > 3 || output.overflow_policy < 0 ||
            output.overflow_policy > 2 ||
            (output.collector_kind == 2 && output.collector_capacity == 0)) {
            throw std::invalid_argument("CppExecutor contract=native_output_descriptor");
        }
    }
    for (std::size_t index = 0; index < source_count_; ++index) {
        if (source_ends_[index] < source_starts_[index] || source_statuses_[index] > 2 ||
            source_resets_[index] > 1) {
            throw std::invalid_argument("CppExecutor contract=stream_item_abi");
        }
    }
}

GraphRuntimeResult GraphRuntimeSession::run(
    bool has_logical_end,
    std::int64_t logical_end_numerator,
    std::int64_t logical_end_denominator
) const {
    GraphRuntimeResult result;
    result.status_counts.assign(3, 0);
    std::int64_t timebase = source_timebase_denominator_;
    for (const GraphNodeSpec& node : nodes_) {
        if (node.opcode == 1) {
            timebase = checked_lcm(timebase, node.period_denominator);
        }
    }
    if (has_logical_end) {
        timebase = checked_lcm(timebase, logical_end_denominator);
    }
    const std::int64_t source_scale = timebase / source_timebase_denominator_;
    const std::int64_t boundary_ticks = has_logical_end
        ? checked_scale(logical_end_numerator, timebase / logical_end_denominator)
        : std::numeric_limits<std::int64_t>::max();

    std::unordered_map<std::int64_t, GraphBatch> batches;
    const auto scheduler_start = std::chrono::steady_clock::now();
    for (const GraphNodeSpec& node : nodes_) {
        GraphBatch batch;
        if (node.opcode == 0) {
            batch.reserve(source_count_);
            std::int64_t segment = 0;
            for (std::size_t index = 0; index < source_count_; ++index) {
                if (source_resets_[index] != 0) {
                    ++segment;
                }
                const std::int64_t start = checked_scale(source_starts_[index], source_scale);
                const std::int64_t end = checked_scale(source_ends_[index], source_scale);
                if (has_logical_end && end > boundary_ticks) {
                    continue;
                }
                GraphItem item;
                const std::size_t offset = index * source_width_;
                item.values.assign(
                    source_values_.begin() + static_cast<std::ptrdiff_t>(offset),
                    source_values_.begin() + static_cast<std::ptrdiff_t>(offset + source_width_)
                );
                item.shape = {source_width_};
                item.start = start;
                item.end = end;
                item.sequence = static_cast<std::int64_t>(batch.size());
                item.status = source_statuses_[index];
                item.provenance = {static_cast<std::int64_t>(index)};
                item.metadata_source_index = static_cast<std::int64_t>(index);
                item.segment = segment;
                batch.push_back(std::move(item));
            }
        } else {
            const auto input = batches.find(node.input_port);
            if (input == batches.end()) {
                throw std::runtime_error("CppExecutor contract=native_input_batch");
            }
            if (node.opcode == 1) {
                batch = run_rate_node(input->second, node, timebase);
            } else if (node.opcode == 2) {
                batch = run_frame_node(input->second, node);
            } else {
                const auto kernel_start = std::chrono::steady_clock::now();
                batch = run_map_node(input->second, node);
                result.kernel_ns += elapsed_ns(kernel_start, std::chrono::steady_clock::now());
            }
        }
        for (const GraphItem& item : batch) {
            result.status_counts[item.status] += 1;
        }
        batches.emplace(node.output_port, std::move(batch));
        ++result.executed_node_count;
    }
    const auto scheduler_end = std::chrono::steady_clock::now();
    result.scheduler_ns = elapsed_ns(scheduler_start, scheduler_end) - result.kernel_ns;

    const auto select_start = std::chrono::steady_clock::now();
    for (const GraphOutputSpec& output_spec : outputs_) {
        const GraphBatch& batch = batches.at(output_spec.port_id);
        RuntimeResult output;
        output.status_counts = result.status_counts;
        output.received_count = batch.size();
        output.timebase_denominator = timebase;
        std::size_t first = 0;
        std::size_t count = 0;
        if (output_spec.collector_kind == 0) {
            output.dropped_count = batch.size();
        } else if (output_spec.collector_kind == 1) {
            count = batch.empty() ? 0 : 1;
            first = batch.empty() ? 0 : batch.size() - 1;
            output.dropped_count = batch.size() - count;
        } else if (output_spec.collector_kind == 3 ||
                   batch.size() <= output_spec.collector_capacity) {
            count = batch.size();
        } else if (output_spec.overflow_policy == 0) {
            output.overflowed = true;
        } else if (output_spec.overflow_policy == 1) {
            count = output_spec.collector_capacity;
            first = batch.size() - count;
            output.dropped_count = batch.size() - count;
        } else {
            count = output_spec.collector_capacity;
            output.dropped_count = batch.size() - count;
        }
        output.retained_count = count;
        std::vector<std::size_t> value_offsets = {0};
        std::vector<std::size_t> shape_offsets = {0};
        std::vector<std::size_t> provenance_offsets = {0};
        std::vector<std::int64_t> invalid_nodes;
        std::vector<std::size_t> invalid_offsets = {0};
        std::vector<std::size_t> shapes;
        std::vector<std::int64_t> metadata_indices;
        if (!output.overflowed) {
            for (std::size_t offset = 0; offset < count; ++offset) {
                const GraphItem& item = batch[first + offset];
                output.values.insert(output.values.end(), item.values.begin(), item.values.end());
                value_offsets.push_back(output.values.size());
                shapes.insert(shapes.end(), item.shape.begin(), item.shape.end());
                shape_offsets.push_back(shapes.size());
                output.sequences.push_back(item.sequence);
                output.starts.push_back(item.start);
                output.ends.push_back(item.end);
                output.statuses.push_back(item.status);
                append_offsets(item.provenance, output.provenance, provenance_offsets);
                append_offsets(item.invalid_nodes, invalid_nodes, invalid_offsets);
                metadata_indices.push_back(item.metadata_source_index);
            }
        }
        output.output_boundary_bytes = checked_size_multiply(
            output.values.size(),
            sizeof(double)
        );
        result.output_boundary_bytes = checked_size_add(
            result.output_boundary_bytes,
            output.output_boundary_bytes
        );
        result.outputs.push_back(std::move(output));
        result.value_offsets.push_back(std::move(value_offsets));
        result.shapes.push_back(std::move(shapes));
        result.shape_offsets.push_back(std::move(shape_offsets));
        result.provenance_offsets.push_back(std::move(provenance_offsets));
        result.invalid_nodes.push_back(std::move(invalid_nodes));
        result.invalid_node_offsets.push_back(std::move(invalid_offsets));
        result.metadata_source_indices.push_back(std::move(metadata_indices));
    }
    result.output_select_ns = elapsed_ns(select_start, std::chrono::steady_clock::now());
    result.owned_input_bytes = 0;
    for (const std::size_t byte_count : {
             checked_size_multiply(source_values_.size(), sizeof(double)),
             checked_size_multiply(source_starts_.size(), sizeof(std::int64_t)),
             checked_size_multiply(source_ends_.size(), sizeof(std::int64_t)),
             checked_size_multiply(source_statuses_.size(), sizeof(std::uint8_t)),
             checked_size_multiply(source_resets_.size(), sizeof(std::uint8_t)),
         }) {
        result.owned_input_bytes = checked_size_add(result.owned_input_bytes, byte_count);
    }
    for (const GraphNodeSpec& node : nodes_) {
        result.owned_input_bytes = checked_size_add(
            result.owned_input_bytes,
            checked_size_multiply(node.kernel_parameters.size(), sizeof(double))
        );
    }
    return result;
}

}  // namespace chronowire::cpp_runtime
