#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace {
    at::Tensor reduce_sum_cpu(const at::Tensor& x) {
        TORCH_CHECK(
        x.device().is_cpu(),
        "reduce_sum: x must be on CPU"
        );
        TORCH_CHECK(
            x.scalar_type() == at::kFloat,
            "reduce_sum: x must be float32"
        );
        TORCH_CHECK(
            x.layout() == at::kStrided,
            "reduce_sum: x must be strided"
        );
        TORCH_CHECK(
            x.is_contiguous(),
            "reduce_sum: x must be contiguous"
        );

        auto out = at::zeros({}, x.options());
        const float* in = x.data_ptr<float>();
        float total = 0.0f;
        for(int64_t i = 0; i < x.numel(); ++i) {
            total += in[i];
        }
        *out.data_ptr<float>() = total;
        return out;
    }

    TORCH_LIBRARY_FRAGMENT(jay_ops, m) {
        m.def("reduce_sum(Tensor x) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(jay_ops, CPU, m) {
        m.impl("reduce_sum", &reduce_sum_cpu);
    }
}