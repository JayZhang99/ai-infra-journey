#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inferflow {

struct TensorInput {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    std::string dtype{"float32"};
};

struct TensorOutput {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

class IRuntime {
public:
    virtual ~IRuntime() = default;
    virtual TensorOutput infer(const TensorInput& input) = 0;
};

}  // namespace inferflow