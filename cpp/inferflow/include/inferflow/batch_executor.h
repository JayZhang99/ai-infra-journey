#pragma once

#include "batch_plan.h"
#include "inferflow/runtime.h"

#include <exception>
#include <optional>
#include <vector>

namespace inferflow {

// Teaching adapter for the future ServiceCore::execute_batch method.
// Call only from the single Worker, after cancellation/deadline filtering.
// Keep ServiceCore::execute_one as the old synchronous contract.
inline std::vector<Result> execute_batch_example(
    IRuntime& runtime,
    const std::vector<RequestContext*>& requests
) {
    std::vector<Result> results(
        requests.size(),
        {StatusCode::kUnavailable, std::nullopt, "not processed"});
    std::vector<const RequestContext*> valid;
    std::vector<std::size_t> positions;

    for (std::size_t i = 0; i < requests.size(); ++i) {
        RequestContext* ctx = requests[i];
        if (!ctx) {
            results[i] = {StatusCode::kInvalidArgument, std::nullopt,
                          "null request"};
            continue;
        }
        if (ctx->result) {
            results[i] = *ctx->result;
            continue;
        }
        try {
            (void)make_batch_plan({ctx});  // validate this request alone
            valid.push_back(ctx);
            positions.push_back(i);
        } catch (const std::invalid_argument& error) {
            ctx->state = RequestState::kInvalid;
            results[i] = {StatusCode::kInvalidArgument,
                          std::nullopt, error.what()};
            ctx->result = results[i];
        }
    }

    if (valid.empty()) return results;

    try {
        BatchPlan plan = make_batch_plan(valid);
        for (std::size_t pos : positions) {
            requests[pos]->state = RequestState::kRunning;
        }
        // Exactly one backend call for all compatible valid requests.
        const TensorOutput merged = runtime.infer(plan.merged);
        auto parts = split_batch_output(plan, merged);
        for (std::size_t j = 0; j < positions.size(); ++j) {
            RequestContext& ctx = *requests[positions[j]];
            ctx.state = RequestState::kSucceeded;
            results[positions[j]] = {
                StatusCode::kOk, std::move(parts[j]), ""};
            ctx.result = results[positions[j]];
        }
    } catch (const std::exception&) {
        // With no per-item backend error protocol, a batch-level failure
        // applies to every valid member of that batch.
        for (std::size_t pos : positions) {
            RequestContext& ctx = *requests[pos];
            ctx.state = RequestState::kRuntimeFailed;
            results[pos] = {StatusCode::kRuntimeError, std::nullopt,
                            "batch runtime failed"};
            ctx.result = results[pos];
        }
    }
    return results;
}

}  // namespace inferflow
