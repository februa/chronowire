#include "cpp_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace chronowire::cpp_runtime {
namespace {

template <typename T>
std::vector<T> copy_buffer(const char* data, std::size_t byte_count, std::size_t count) {
    if (byte_count != count * sizeof(T)) {
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
        source_count * source_width
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
        beam_count * weight_channel_count
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
    result.output_width = beam_count_ * frame_size_;
    result.owned_input_bytes =
        source_values_.size() * sizeof(double) +
        source_starts_.size() * sizeof(std::int64_t) +
        source_ends_.size() * sizeof(std::int64_t) +
        source_statuses_.size() * sizeof(std::uint8_t) +
        weights_.size() * sizeof(double);

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
    std::vector<double> frame_values(frame_count * frame_size_ * source_width_);
    std::vector<std::int64_t> frame_starts(frame_count);
    std::vector<std::int64_t> frame_ends(frame_count);
    std::vector<std::uint8_t> frame_statuses(frame_count);
    std::vector<std::int64_t> frame_provenance(frame_count * frame_size_);
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
    std::vector<double> all_outputs(frame_count * result.output_width);
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
        const std::size_t value_begin = first_retained * result.output_width;
        const std::size_t value_end = value_begin + retained_count * result.output_width;
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
    result.output_boundary_bytes = result.values.size() * sizeof(double);
    const auto select_end = std::chrono::steady_clock::now();
    result.scheduler_ns = elapsed_ns(scheduler_start, scheduler_end);
    result.kernel_ns = elapsed_ns(kernel_start, kernel_end);
    result.output_select_ns = elapsed_ns(select_start, select_end);
    return result;
}

}  // namespace chronowire::cpp_runtime
