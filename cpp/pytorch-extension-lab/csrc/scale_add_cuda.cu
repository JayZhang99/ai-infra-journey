#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace {

__global__ void scale_add_kernel(
    const float* input,
    float* output,
    int64_t n,
    float alpha
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(index < n) {
        output[index] = input[index] * alpha + 1.0f;
    }

}

at::Tensor scale_add_cuda(
    const at::Tensor& x,
    double alpha
) {

    TORCH_CHECK(
        x.device().is_cuda(),
        "scale_add: x must be on CUDA"
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

    c10::cuda::CUDAGuard device_guard(x.device());

    auto out = at::empty_like(x);   // 让输出shape/dtype/device 与输入一致
    const int64_t n = x.numel();

    if (n==0) {
        return out;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads -1)/ threads);


    cudaStream_t stream = 
        at::cuda::getCurrentCUDAStream(
            x.get_device()
        );
    scale_add_kernel<<<
        blocks, threads, 0, stream
        >>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            n,
            static_cast<float>(alpha)
        );
    
    C1O_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

TORCH_LIBRARY_IMPL(jay_ops, CUDA, m) {
    m.impl(
        "scale_add",
        &scale_add_cuda
    );
}

}




