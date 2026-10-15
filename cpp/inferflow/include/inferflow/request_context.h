#pragma once

#include "inferflow/runtime.h"

#include <chrono>
#include <optional>
#include <string>

namespace inferflow {

struct RequestTiming {
    using Clock = std::chrono::steady_clock;

    Clock::time_point submitted_at{};
    Clock::time_point enqueued_at{};
    Clock::time_point dequeued_at{};
    Clock::time_point bucketed_at{};
    Clock::time_point dispatched_at{};
    Clock::time_point runtime_begin{};
    Clock::time_point runtime_end{};
    Clock::time_point completed_at{};
};

struct TimingMs {
    double queue_time{0.0};
    double pre_bucket_wait{0.0};
    double batch_wait{0.0};
    double runtime_time{0.0};
    double end_to_end{0.0};
    // No transport serialization exists in the current in-process engine.
    std::optional<double> serialization_time;
};

enum class StatusCode {
    kOk,
    kInvalidArgument,
    kRuntimeError,
    kUnavailable,
    kResourceExhausted,
    kDeadlineExceeded,
    kCancelled,
};

enum class RequestState {
    kAccepted,
    kValidated,
    kRunning,
    kSucceeded,
    kInvalid,
    kRuntimeFailed,
    kUnavailable,
    kQueued,
    kExpired,
    kCancelled,
    kRejectedFull,
};

struct Result {
    StatusCode code;
    std::optional<TensorOutput> output;
    std::string message;
    TimingMs timing;
};

struct RequestContext {
    std::string request_id;
    TensorInput input;
    RequestState state{RequestState::kAccepted};
    std::optional<Result> result;
    RequestTiming timing;
};

}  // namespace inferflow
