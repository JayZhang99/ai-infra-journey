#pragma once

#include "inferflow/runtime.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace inferflow {

struct RuntimeStats {
    int constructed{0};
    int calls{0};
    int destroyed{0};
};

class FakeRuntime final : public IRuntime {
public:
    explicit FakeRuntime(
        std::shared_ptr<RuntimeStats> stats,
        bool throw_on_call = false
    )
        : stats_(std::move(stats)),
          throw_on_call_(throw_on_call) {
        ++stats_->constructed;
    }

    ~FakeRuntime() override {
        ++stats_->destroyed;
    }

    TensorOutput infer(const TensorInput& input) override {
        ++stats_->calls;

        if (throw_on_call_) {
            throw std::runtime_error("simulated runtime failure");
        }

        return {input.shape, input.values};
    }

private:
    std::shared_ptr<RuntimeStats> stats_;
    bool throw_on_call_;
};

}  // namespace inferflow