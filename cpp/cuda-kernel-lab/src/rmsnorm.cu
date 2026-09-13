#include "rmsnorm.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cmath>

namespace {
    void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}
}

const char* rmsnorm_kind_name(RmsNormKind kind) {
    switch (kind) {
        case RmsNormKind::Baseline:
            return "baseline";

        case RmsNormKind::Block:
            return "block";
    }

    return "unknown";
}

__global__ void rmsnorm_baseline_f32(
    const float* x,
    const float* weight,
    float* y,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in = x + row * cols;
    float* out = y + row * cols;

    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = in[c];
        sum_sq += v * v;
    }

    float mean_sq = sum_sq / static_cast<float>(cols);
    float inv_rms = rsqrtf(mean_sq + eps);

    for(int c = 0; c < cols; ++c) {
        out[c] = in[c] * inv_rms * weight[c];
    }
}

__device__ __forceinline__
float warp_sum(float value, unsigned mask) {
    for (int offset = 16;
         offset > 0;
         offset >>= 1) {
            value += __shfl_down_sync(
                mask,
                value,
                offset
            );
         }
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float warp_values[32];
    __shared__ float result;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    value = warp_sum(value, __activemask());
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < nwarps
                ? warp_values[lane]
                : 0.0f;
        value = warp_sum(value, __activemask());

        if(lane == 0) result = value;
    }
    
    __syncthreads();
    return result;
}

__global__ void rmsnorm_block_f32(
    const float* x,
    const float* weight,
    float* y,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if(row >= rows) return;

    const float* in = row * cols + x;
    float* out = y + row  * cols;

    float local_sq = 0.0f;
    for(int c = tid; c< cols; c += blockDim.x) {
        float v = in[c];
        local_sq += v * v;
    }

    float sum_sq = block_sum(local_sq);

    float mean_sq = sum_sq / static_cast<float>(cols);

    float inv_ms = rsqrtf(mean_sq + eps);

    for (int c = tid; c < cols; c += blockDim.x) {
        out[c] = in[c] * inv_ms * weight[c];
    }

}

void launch_rmsnorm_f32(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps,
    int threads,
    RmsNormKind kind,
    cudaStream_t stream
) {
    if (input == nullptr ||
        output == nullptr ||
        weight == nullptr) {
        throw std::invalid_argument(
            "device pointers and weight pointers must not be null"
        );
    }

    if (rows <= 0 || cols <=0) {
        throw std::invalid_argument(
            "rows and cols must be positvie"
        );
    }

    if (eps <= 0 || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "eps must be positive and finitie"
        );
    }

    const bool is_multy_of_32 = threads >0 && threads % 32 ==0;
    if(!is_multy_of_32 || threads < 32 || threads > 1024) {
        throw std::invalid_argument(
            "threads must be a multiple of 32 "
            "between 32 and 1024"
        );
    }

    switch (kind) {
        case RmsNormKind::Baseline: {
            int blocks = (rows + threads -1) / threads;
            rmsnorm_baseline_f32<<<
            blocks,
            threads,
            0,
            stream
            >>>(
                input,
                weight,
                output,
                rows,
                cols,
                eps
            );
            break;
        }

        case RmsNormKind::Block: {
            rmsnorm_block_f32<<<
            rows,
            threads,
            0,
            stream
            >>>(
                input,
                weight,
                output,
                rows,
                cols,
                eps
            );
            break;
        }
            

        default:
            throw std::invalid_argument(
                "unsupported RmsNormKind"
            );
    }

    cuda_check(
        cudaGetLastError(),
        "launch rmsnorm kernel"
    );
}
