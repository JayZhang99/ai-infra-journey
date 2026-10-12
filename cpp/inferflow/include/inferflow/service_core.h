#pragma once

#include "inferflow/request_context.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace inferflow {

class ServiceCore {
public:
    explicit ServiceCore(std::unique_ptr<IRuntime> runtime);

    Result execute_one(RequestContext& ctx);
    // Results correspond to requests by index. The caller owns the contexts.
    std::vector<Result> execute_batch(
        const std::vector<RequestContext*>& requests);
    void stop();

private:
    static std::optional<std::string> validate(const TensorInput& input);

    static Result finish(
        RequestContext& ctx,
        RequestState final_state,
        Result result
    );

    std::unique_ptr<IRuntime> runtime_;
    bool stopping_{false};
};

}  // namespace inferflow
