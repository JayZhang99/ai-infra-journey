#include "inferflow/service_core.h"
#include "inferflow/batch_plan.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inferflow {

ServiceCore::ServiceCore(std::unique_ptr<IRuntime> runtime)
    : runtime_(std::move(runtime)) {
    if (!runtime_) {
        throw std::invalid_argument("runtime must not be null");
    }
}

std::optional<std::string> ServiceCore::validate(
    const TensorInput& input
) {
    if (input.dtype != "float32") {
        return "only float32 is supported";
    }
    if (input.shape.size() != 3) {
        return "expected rank-3 input";
    }
    if (input.shape.back() != 8) {
        return "hidden dimension must be 8";
    }

    std::size_t count = 1;
    for (std::int64_t dim : input.shape) {
        if (dim <= 0) {
            return "shape dimensions must be positive";
        }

        const auto size = static_cast<std::size_t>(dim);
        if (size > std::numeric_limits<std::size_t>::max() / count) {
            return "shape element count overflow";
        }
        count *= size;
    }

    if (count != input.values.size()) {
        return "shape/data size mismatch";
    }
    return std::nullopt;
}

Result ServiceCore::finish(
    RequestContext& ctx,
    RequestState final_state,
    Result result
) {
    ctx.state = final_state;
    ctx.result = std::move(result);
    return *ctx.result;
}

Result ServiceCore::execute_one(RequestContext& ctx) {
    // 已完成的请求返回原结果，不再执行第二次推理。
    if (ctx.result) {
        return *ctx.result;
    }

    if (stopping_) {
        return finish(
            ctx,
            RequestState::kUnavailable,
            {StatusCode::kUnavailable, std::nullopt, "service is stopping"}
        );
    }

    if (ctx.request_id.empty()) {
        return finish(
            ctx,
            RequestState::kInvalid,
            {StatusCode::kInvalidArgument, std::nullopt,
             "request_id is empty"}
        );
    }

    if (auto error = validate(ctx.input)) {
        return finish(
            ctx,
            RequestState::kInvalid,
            {StatusCode::kInvalidArgument, std::nullopt, *error}
        );
    }

    ctx.state = RequestState::kValidated;
    ctx.state = RequestState::kRunning;

    try {
        TensorOutput output = runtime_->infer(ctx.input);
        return finish(
            ctx,
            RequestState::kSucceeded,
            {StatusCode::kOk, std::move(output), ""}
        );
    } catch (const std::exception&) {
        return finish(
            ctx,
            RequestState::kRuntimeFailed,
            {StatusCode::kRuntimeError, std::nullopt,
             "runtime execution failed"}
        );
    }
}

std::vector<Result> ServiceCore::execute_batch(
    const std::vector<RequestContext*>& requests) {
    std::vector<Result> results(requests.size(),
        {StatusCode::kInvalidArgument, std::nullopt, "null request"});
    std::vector<const RequestContext*> valid;
    std::vector<std::size_t> positions;

    for (std::size_t i = 0; i < requests.size(); ++i) {
        RequestContext* ctx = requests[i];
        if (!ctx) continue;
        if (ctx->result) {
            results[i] = *ctx->result;
        } else if (stopping_) {
            results[i] = finish(*ctx, RequestState::kUnavailable,
                {StatusCode::kUnavailable, std::nullopt,
                 "service is stopping"});
        } else if (ctx->request_id.empty()) {
            results[i] = finish(*ctx, RequestState::kInvalid,
                {StatusCode::kInvalidArgument, std::nullopt,
                 "request_id is empty"});
        } else if (auto error = validate(ctx->input)) {
            results[i] = finish(*ctx, RequestState::kInvalid,
                {StatusCode::kInvalidArgument, std::nullopt, *error});
        } else {
            ctx->state = RequestState::kValidated;
            valid.push_back(ctx);
            positions.push_back(i);
        }
    }
    if (valid.empty()) return results;

    // The engine groups by sequence length. Direct callers with mixed valid
    // shapes keep the old per-request behavior rather than losing requests.
    const auto sequence = valid.front()->input.shape[1];
    bool compatible = true;
    for (const RequestContext* ctx : valid) {
        if (ctx->input.shape[1] != sequence) compatible = false;
    }
    if (!compatible) {
        for (std::size_t pos : positions) {
            results[pos] = execute_one(*requests[pos]);
        }
        return results;
    }

    try {
        BatchPlan plan = make_batch_plan(valid);
        for (std::size_t pos : positions) {
            requests[pos]->state = RequestState::kRunning;
        }
        TensorOutput merged = runtime_->infer(plan.merged);
        auto parts = split_batch_output(plan, merged);
        for (std::size_t j = 0; j < positions.size(); ++j) {
            results[positions[j]] = finish(*requests[positions[j]],
                RequestState::kSucceeded,
                {StatusCode::kOk, std::move(parts[j]), ""});
        }
    } catch (const std::exception&) {
        for (std::size_t pos : positions) {
            results[pos] = finish(*requests[pos],
                RequestState::kRuntimeFailed,
                {StatusCode::kRuntimeError, std::nullopt,
                 "batch runtime execution failed"});
        }
    }
    return results;
}

void ServiceCore::stop() {
    stopping_ = true;
}

}  // namespace inferflow
