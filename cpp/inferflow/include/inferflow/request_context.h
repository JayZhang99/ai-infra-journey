#pragma once

#include "inferflow/runtime.h"

#include <optional>
#include <string>

namespace inferflow {

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
};

struct RequestContext {
    std::string request_id;
    TensorInput input;
    RequestState state{RequestState::kAccepted};
    std::optional<Result> result;
};

}  // namespace inferflow