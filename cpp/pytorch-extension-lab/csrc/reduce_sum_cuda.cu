#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

#include "cuda-kernel-lab/include/reduction.cuh"

namespace {
    at::Tensor reduce_sum_cuda(const at::Tensor& x) {
        TORCH_CHECK(
        x.is_cuda(),
            "reduce_sum: x must be on CUDA"
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

        c10::cuda::CUDAGuard guard(x.device());
        auto out = at::zeros({}, x.options());
        const int64_t n = x.numel();
        if(n == 0) return out;

        const int threads = 256;
        const int64_t needed = (n + threads -1) / threads;
        const int blocks = static_cast<int>(std::min<int64_t>(needed, 1024));

        cudaStream_t stream = 
            at::cuda::getCurrentCUDAStream(
                x.get_device()
            );
        
        launch_reduction(
            x.data_ptr<float>(),
            static_cast<size_t>(n),
            out.data_ptr<float>(),
            blocks,
            threads,
            ReduceKind::Warp,
            stream
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    TORCH_LIBRARY_IMPL(jay_ops, CUDA, m) {
        m.impl(
            "reduce_sum",
            &reduce_sum_cuda
        );
    }
 }