#include "cpp_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
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
    std::vector<std::int64_t> degraded_nodes;
    struct NativeDiagnostic {
        std::uint8_t severity = 0;
        std::int64_t node_id = -1;
        std::string code;
        std::string message;
    };
    std::vector<NativeDiagnostic> native_diagnostics;
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

enum class GraphKernelKind {
    identity_f64,
    fixed_cbf_f64,
    covariance_f64,
    mvdr_weights_f64,
    apply_weights_f64,
    external_operation,
};

GraphKernelKind resolve_graph_kernel(const GraphNodeSpec& node) {
    const bool has_external_entry =
        node.native_create != 0 && node.native_process != 0 && node.native_destroy != 0;
    if (has_external_entry) {
        return GraphKernelKind::external_operation;
    }
    if (node.native_create != 0 || node.native_process != 0 || node.native_flush != 0 ||
        node.native_destroy != 0) {
        throw std::invalid_argument("CppExecutor contract=native_module_function_table");
    }
    if (node.kernel_abi == "chronowire.kernel.identity_f64.v1" &&
        node.process_model == "identity_f64") {
        return GraphKernelKind::identity_f64;
    }
    if (node.kernel_abi == "chronowire.reference.fixed_cbf_f64.v1" &&
        node.process_model == "fixed_cbf_f64_frame") {
        return GraphKernelKind::fixed_cbf_f64;
    }
    if (node.kernel_abi == "chronowire.reference.covariance_accumulator_f64_frame.v1" &&
        node.process_model == "covariance_accumulator_f64_frame") {
        return GraphKernelKind::covariance_f64;
    }
    if (node.kernel_abi == "chronowire.reference.mvdr_weights_f64.v1" &&
        node.process_model == "mvdr_weights_f64") {
        return GraphKernelKind::mvdr_weights_f64;
    }
    if (node.kernel_abi == "chronowire.reference.apply_weights_f64_latest.v1" &&
        node.process_model == "apply_weights_f64_latest") {
        return GraphKernelKind::apply_weights_f64;
    }
    throw std::invalid_argument("CppExecutor contract=kernel_abi_table");
}

class ExternalKernelState {
public:
    explicit ExternalKernelState(const GraphNodeSpec& node) : node_(node) {
        create_ = reinterpret_cast<CwCreateFnV1>(node.native_create);
        process_ = reinterpret_cast<CwProcessFnV1>(node.native_process);
        destroy_ = reinterpret_cast<CwDestroyFnV1>(node.native_destroy);
        std::array<char, 512> error{};
        session_ = create_(
            node.kernel_parameters.empty() ? nullptr : node.kernel_parameters.data(),
            node.kernel_parameters.size(),
            error.data(),
            error.size()
        );
        if (session_ == nullptr) {
            throw std::runtime_error(
                std::string("CppExecutor contract=native_module_create node=") +
                std::to_string(node.node_id) + " port=" + std::to_string(node.output_port) +
                " error=" + error.data()
            );
        }
    }

    ExternalKernelState(const ExternalKernelState&) = delete;
    ExternalKernelState& operator=(const ExternalKernelState&) = delete;

    ~ExternalKernelState() {
        if (session_ != nullptr && destroy_ != nullptr) {
            destroy_(session_);
        }
    }

    void process(const std::vector<const GraphItem*>& inputs, GraphItem& output) const {
        std::vector<CwBufferViewV1> views;
        views.reserve(inputs.size());
        for (const GraphItem* input : inputs) {
            views.push_back(CwBufferViewV1{
                input->values.empty() ? nullptr : input->values.data(),
                input->values.size(),
                input->shape.empty() ? nullptr : input->shape.data(),
                input->shape.size(),
            });
        }
        const std::size_t output_count = checked_shape_product(node_.output_shape);
        output.values.assign(output_count, 0.0);
        output.shape = node_.output_shape;
        CwMutableBufferViewV1 output_view{
            output.values.empty() ? nullptr : output.values.data(),
            output.values.size(),
            output.shape.empty() ? nullptr : output.shape.data(),
            output.shape.size(),
        };
        CwProcessResultV1 process_result{};
        std::array<char, 512> error{};
        const int status = process_(
            session_,
            views.data(),
            views.size(),
            &output_view,
            &process_result,
            error.data(),
            error.size()
        );
        if (status != 0) {
            throw std::runtime_error(
                std::string("CppExecutor contract=native_module_process node=") +
                std::to_string(node_.node_id) + " port=" +
                std::to_string(node_.output_port) + " error=" + error.data()
            );
        }
        if (process_result.output_count != output_count || process_result.status > 2 ||
            process_result.diagnostic_severity > 2) {
            throw std::runtime_error(
                std::string("CppExecutor contract=native_module_result node=") +
                std::to_string(node_.node_id) + " port=" +
                std::to_string(node_.output_port)
            );
        }
        output.status = std::max(output.status, process_result.status);
        const bool has_code = process_result.diagnostic_code != nullptr &&
            process_result.diagnostic_code[0] != '\0';
        const bool has_message = process_result.diagnostic_message != nullptr &&
            process_result.diagnostic_message[0] != '\0';
        if (has_code != has_message) {
            throw std::runtime_error(
                std::string("CppExecutor contract=native_module_diagnostic node=") +
                std::to_string(node_.node_id) + " port=" +
                std::to_string(node_.output_port)
            );
        }
        if (has_code) {
            output.native_diagnostics.push_back(GraphItem::NativeDiagnostic{
                process_result.diagnostic_severity,
                node_.node_id,
                process_result.diagnostic_code,
                process_result.diagnostic_message,
            });
        }
    }

private:
    const GraphNodeSpec& node_;
    CwCreateFnV1 create_ = nullptr;
    CwProcessFnV1 process_ = nullptr;
    CwDestroyFnV1 destroy_ = nullptr;
    void* session_ = nullptr;
};

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
    if (node.rate_policy == 1) {
        for (const GraphItem& item : input) {
            if (item.start % period_ticks != 0) {
                continue;
            }
            GraphItem emitted = item;
            emitted.sequence = static_cast<std::int64_t>(output.size());
            output.push_back(std::move(emitted));
        }
        return output;
    }
    if (node.rate_policy != 0) {
        throw std::invalid_argument("CppExecutor contract=rate_policy");
    }
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
                frame.degraded_nodes.insert(
                    frame.degraded_nodes.end(),
                    item.degraded_nodes.begin(),
                    item.degraded_nodes.end()
                );
                frame.native_diagnostics.insert(
                    frame.native_diagnostics.end(),
                    item.native_diagnostics.begin(),
                    item.native_diagnostics.end()
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

std::vector<double> solve_linear_system(
    const std::vector<double>& matrix,
    const std::vector<double>& right
) {
    const std::size_t size = right.size();
    if (matrix.size() != checked_size_multiply(size, size) || size == 0) {
        throw std::invalid_argument("CppExecutor contract=mvdr_linear_shape");
    }
    std::vector<double> rows(checked_size_multiply(size, size + 1), 0.0);
    for (std::size_t row = 0; row < size; ++row) {
        std::copy_n(
            matrix.begin() + static_cast<std::ptrdiff_t>(row * size),
            size,
            rows.begin() + static_cast<std::ptrdiff_t>(row * (size + 1))
        );
        rows[row * (size + 1) + size] = right[row];
    }
    for (std::size_t pivot = 0; pivot < size; ++pivot) {
        std::size_t selected = pivot;
        for (std::size_t row = pivot + 1; row < size; ++row) {
            if (std::abs(rows[row * (size + 1) + pivot]) >
                std::abs(rows[selected * (size + 1) + pivot])) {
                selected = row;
            }
        }
        if (std::abs(rows[selected * (size + 1) + pivot]) <= 1.0e-12) {
            throw std::invalid_argument("CppExecutor contract=mvdr_singular_covariance");
        }
        for (std::size_t column = pivot; column <= size; ++column) {
            std::swap(
                rows[pivot * (size + 1) + column],
                rows[selected * (size + 1) + column]
            );
        }
        const double divisor = rows[pivot * (size + 1) + pivot];
        for (std::size_t column = pivot; column <= size; ++column) {
            rows[pivot * (size + 1) + column] /= divisor;
        }
        for (std::size_t row = 0; row < size; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = rows[row * (size + 1) + pivot];
            for (std::size_t column = pivot; column <= size; ++column) {
                rows[row * (size + 1) + column] -=
                    factor * rows[pivot * (size + 1) + column];
            }
        }
    }
    std::vector<double> result(size);
    for (std::size_t row = 0; row < size; ++row) {
        result[row] = rows[row * (size + 1) + size];
    }
    return result;
}

GraphBatch run_map_node(
    const std::vector<const GraphBatch*>& inputs,
    const GraphNodeSpec& node
) {
    if (inputs.empty() || inputs.size() != node.input_semantics.size()) {
        throw std::invalid_argument("CppExecutor contract=map_input_descriptor");
    }
    const GraphKernelKind kind = resolve_graph_kernel(node);
    const GraphBatch& input = *inputs[0];
    GraphBatch output;
    output.reserve(input.size());
    std::vector<std::size_t> latest_indices(inputs.size(), 0);
    std::vector<bool> has_latest(inputs.size(), false);
    std::vector<double> covariance_sums;
    std::size_t covariance_sample_count = 0;
    std::unique_ptr<ExternalKernelState> external_session;
    if (kind == GraphKernelKind::external_operation) {
        external_session = std::make_unique<ExternalKernelState>(node);
    }
    for (const GraphItem& item : input) {
        GraphItem mapped = item;
        mapped.sequence = static_cast<std::int64_t>(output.size());
        std::vector<const GraphItem*> selected_inputs = {&item};
        bool missing = false;
        for (std::size_t input_index = 1; input_index < inputs.size(); ++input_index) {
            if (node.input_semantics[input_index] != 1) {
                throw std::invalid_argument("CppExecutor contract=map_secondary_latest");
            }
            const GraphBatch& candidates = *inputs[input_index];
            while (latest_indices[input_index] < candidates.size() &&
                   candidates[latest_indices[input_index]].start <= item.start) {
                has_latest[input_index] = true;
                ++latest_indices[input_index];
            }
            if (!has_latest[input_index]) {
                missing = true;
                break;
            }
            const GraphItem& latest = candidates[latest_indices[input_index] - 1];
            selected_inputs.push_back(&latest);
            mapped.status = std::max(mapped.status, latest.status);
            mapped.provenance.insert(
                mapped.provenance.end(), latest.provenance.begin(), latest.provenance.end()
            );
            mapped.invalid_nodes.insert(
                mapped.invalid_nodes.end(), latest.invalid_nodes.begin(), latest.invalid_nodes.end()
            );
            mapped.degraded_nodes.insert(
                mapped.degraded_nodes.end(),
                latest.degraded_nodes.begin(),
                latest.degraded_nodes.end()
            );
            mapped.native_diagnostics.insert(
                mapped.native_diagnostics.end(),
                latest.native_diagnostics.begin(),
                latest.native_diagnostics.end()
            );
        }
        if (missing) {
            continue;
        }
        if (mapped.status == 2 && !node.accepts_invalid) {
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
        if (kind == GraphKernelKind::external_operation) {
            external_session->process(selected_inputs, mapped);
            output.push_back(std::move(mapped));
            continue;
        }
        if (kind == GraphKernelKind::covariance_f64) {
            if (item.shape.size() != 2 || node.output_shape.size() != 2 ||
                node.kernel_parameters.size() != 1 || node.parameter_shape != std::vector<std::size_t>({1})) {
                throw std::invalid_argument("CppExecutor contract=covariance_shape");
            }
            const std::size_t sample_count = item.shape[0];
            const std::size_t channel_count = item.shape[1];
            if (node.output_shape[0] != channel_count || node.output_shape[1] != channel_count) {
                throw std::invalid_argument("CppExecutor contract=covariance_output_shape");
            }
            if (covariance_sums.empty()) {
                covariance_sums.assign(
                    checked_size_multiply(channel_count, channel_count),
                    0.0
                );
            }
            covariance_sample_count = checked_size_add(covariance_sample_count, sample_count);
            mapped.values.assign(checked_size_multiply(channel_count, channel_count), 0.0);
            mapped.shape = node.output_shape;
            for (std::size_t row = 0; row < channel_count; ++row) {
                for (std::size_t column = 0; column < channel_count; ++column) {
                    for (std::size_t sample = 0; sample < sample_count; ++sample) {
                        covariance_sums[row * channel_count + column] +=
                            item.values[sample * channel_count + row] *
                            item.values[sample * channel_count + column];
                    }
                    mapped.values[row * channel_count + column] =
                        covariance_sums[row * channel_count + column] /
                        static_cast<double>(covariance_sample_count) +
                        (row == column ? node.kernel_parameters[0] : 0.0);
                }
            }
            if (covariance_sample_count < checked_size_multiply(channel_count, channel_count)) {
                mapped.status = std::max<std::uint8_t>(mapped.status, 1);
                mapped.degraded_nodes.push_back(node.node_id);
            }
            output.push_back(std::move(mapped));
            continue;
        }
        if (kind == GraphKernelKind::mvdr_weights_f64) {
            if (item.shape.size() != 2 || item.shape[0] != item.shape[1] ||
                node.output_shape.size() != 1 || node.parameter_shape.size() != 1) {
                throw std::invalid_argument("CppExecutor contract=mvdr_weights_shape");
            }
            const std::size_t channel_count = item.shape[0];
            if (node.output_shape[0] != channel_count ||
                node.parameter_shape[0] != channel_count ||
                node.kernel_parameters.size() != channel_count) {
                throw std::invalid_argument("CppExecutor contract=mvdr_steering_shape");
            }
            mapped.values = solve_linear_system(item.values, node.kernel_parameters);
            const double denominator = std::inner_product(
                node.kernel_parameters.begin(),
                node.kernel_parameters.end(),
                mapped.values.begin(),
                0.0
            );
            if (std::abs(denominator) <= 1.0e-12) {
                throw std::invalid_argument("CppExecutor contract=mvdr_denominator");
            }
            for (double& value : mapped.values) {
                value /= denominator;
            }
            mapped.shape = node.output_shape;
            output.push_back(std::move(mapped));
            continue;
        }
        if (kind == GraphKernelKind::apply_weights_f64) {
            if (selected_inputs.size() != 2 || item.shape.size() != 2 ||
                selected_inputs[1]->shape.size() != 1 || node.output_shape.size() != 1) {
                throw std::invalid_argument("CppExecutor contract=apply_weights_shape");
            }
            const std::size_t sample_count = item.shape[0];
            const std::size_t channel_count = item.shape[1];
            const GraphItem& weights = *selected_inputs[1];
            if (weights.shape[0] != channel_count || node.output_shape[0] != sample_count) {
                throw std::invalid_argument("CppExecutor contract=apply_weights_channel_shape");
            }
            mapped.values.assign(sample_count, 0.0);
            mapped.shape = node.output_shape;
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                for (std::size_t channel = 0; channel < channel_count; ++channel) {
                    mapped.values[sample] += item.values[sample * channel_count + channel] *
                        weights.values[channel];
                }
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
    if ((schema_version != "0.3" && schema_version != "0.4") ||
        nodes_.empty() || outputs_.empty()) {
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
            if (!node.output_shape.empty() &&
                checked_shape_product(node.output_shape) != source_width_) {
                throw std::invalid_argument("CppExecutor contract=source_output_shape");
            }
        } else {
            if (node.input_ports.empty() ||
                node.input_ports.size() != node.input_semantics.size()) {
                throw std::invalid_argument("CppExecutor contract=native_input_descriptor");
            }
            for (const std::int64_t input_port : node.input_ports) {
                if (produced_ports.find(input_port) == produced_ports.end()) {
                    throw std::invalid_argument("CppExecutor contract=topological_native_edge");
                }
            }
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
                // 合成Stage ingressでもresolved item shapeを失わない。旧IRだけ幅一次元へ戻す。
                item.shape = node.output_shape.empty()
                    ? std::vector<std::size_t>{source_width_}
                    : node.output_shape;
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
            std::vector<const GraphBatch*> inputs;
            inputs.reserve(node.input_ports.size());
            for (const std::int64_t input_port : node.input_ports) {
                const auto input = batches.find(input_port);
                if (input == batches.end()) {
                    throw std::runtime_error("CppExecutor contract=native_input_batch");
                }
                inputs.push_back(&input->second);
            }
            if (node.opcode == 1) {
                if (inputs.size() != 1) {
                    throw std::runtime_error("CppExecutor contract=rate_input_count");
                }
                batch = run_rate_node(*inputs[0], node, timebase);
            } else if (node.opcode == 2) {
                if (inputs.size() != 1) {
                    throw std::runtime_error("CppExecutor contract=frame_input_count");
                }
                batch = run_frame_node(*inputs[0], node);
            } else {
                const auto kernel_start = std::chrono::steady_clock::now();
                batch = run_map_node(inputs, node);
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
        std::vector<std::int64_t> degraded_nodes;
        std::vector<std::size_t> degraded_offsets = {0};
        std::vector<std::int64_t> native_diagnostic_nodes;
        std::vector<std::uint8_t> native_diagnostic_severities;
        std::vector<std::string> native_diagnostic_codes;
        std::vector<std::string> native_diagnostic_messages;
        std::vector<std::size_t> native_diagnostic_offsets = {0};
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
                append_offsets(item.degraded_nodes, degraded_nodes, degraded_offsets);
                for (const GraphItem::NativeDiagnostic& diagnostic : item.native_diagnostics) {
                    native_diagnostic_nodes.push_back(diagnostic.node_id);
                    native_diagnostic_severities.push_back(diagnostic.severity);
                    native_diagnostic_codes.push_back(diagnostic.code);
                    native_diagnostic_messages.push_back(diagnostic.message);
                }
                native_diagnostic_offsets.push_back(native_diagnostic_nodes.size());
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
        result.degraded_nodes.push_back(std::move(degraded_nodes));
        result.degraded_node_offsets.push_back(std::move(degraded_offsets));
        result.native_diagnostic_nodes.push_back(std::move(native_diagnostic_nodes));
        result.native_diagnostic_severities.push_back(std::move(native_diagnostic_severities));
        result.native_diagnostic_codes.push_back(std::move(native_diagnostic_codes));
        result.native_diagnostic_messages.push_back(std::move(native_diagnostic_messages));
        result.native_diagnostic_offsets.push_back(std::move(native_diagnostic_offsets));
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

CooperativeStageSession::CooperativeStageSession(
    const std::vector<std::int64_t>& stage_ids
) : stage_ids_(stage_ids) {
    if (stage_ids_.empty()) {
        throw std::invalid_argument("CppExecutor contract=nonempty_python_stage_sequence");
    }
    std::unordered_map<std::int64_t, bool> seen;
    for (const std::int64_t stage_id : stage_ids_) {
        if (stage_id < 0 || seen.find(stage_id) != seen.end()) {
            throw std::invalid_argument("CppExecutor contract=unique_python_stage_id");
        }
        seen.emplace(stage_id, true);
    }
}

std::pair<int, std::int64_t> CooperativeStageSession::advance() {
    if (aborted_) {
        throw std::runtime_error("CppExecutor contract=python_stage_session_aborted");
    }
    if (waiting_stage_id_ >= 0) {
        throw std::runtime_error("CppExecutor contract=python_stage_resume_required");
    }
    if (cursor_ >= stage_ids_.size()) {
        return {1, -1};
    }
    waiting_stage_id_ = stage_ids_[cursor_];
    return {0, waiting_stage_id_};
}

void CooperativeStageSession::resume(std::int64_t stage_id) {
    if (aborted_) {
        throw std::runtime_error("CppExecutor contract=python_stage_session_aborted");
    }
    if (waiting_stage_id_ < 0 || waiting_stage_id_ != stage_id) {
        throw std::invalid_argument("CppExecutor contract=python_stage_resume_id");
    }
    waiting_stage_id_ = -1;
    ++cursor_;
}

void CooperativeStageSession::abort() {
    aborted_ = true;
    waiting_stage_id_ = -1;
}

}  // namespace chronowire::cpp_runtime
