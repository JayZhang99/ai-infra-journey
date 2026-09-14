#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace {

at::Tensor scale_add_cpu(
    const at::Tensor& x,
    double alpha
) {

    TORCH_CHECK(
        x.device().is_cpu(),
        "scale_add: x must be on CPU"
    );
    TORCH_CHECK(
        x.scalar_type() == at::kFloat,
        "scale_add: x must be float32"
    );
    TORCH_CHECK(
        x.layout() == at::kStrided,
        "scale_add: x must be strided"
    );
    TORCH_CHECK(
        x.is_contiguous(),
        "scale_add: x must be contiguous"
    );
    TORCH_CHECK(
        std::isfinite(alpha),
        "scale_add: alpha must be finite"
    );

    auto out = at::empty_like(x);   // 让输出shape/dtype/device 与输入一致

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = x.numel();

    for (int64_t i = 0; i < n ; ++i) {
        out_ptr[i] = 
            x_ptr[i] * static_cast<float>(alpha)
            + 1.0f;
    }

    return out;
}

TORCH_LIBRARY(jay_ops, m) {
    m.def(
        "scale_add(Tensor x, float alpha) -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(jay_ops, CPU, m) {
    m.impl(
        "scale_add",
        &scale_add_cpu
    );
}
}   // namespace