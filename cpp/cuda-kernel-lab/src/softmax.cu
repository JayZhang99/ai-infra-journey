#include "softmax.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

__global__ void softmax_baseline_f32(
    const float* x, float* y,
    int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in = x + row * cols;
    float* out = y + row * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) {
        m = fmaxf(m, in[c]);
    }
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) {
        s += expf(in[c]-m);
    }
    for(int c = 0; c < cols; ++c) {
        out[c] = expf(in[c]-m)/s;
    }
}

__device__ float warp_max(float v, unsigned mask) {
    for(int off = 16; off >0; off /= 2) {
        float o = __shfl_down_sync(mask, v, off);
        v = fmaxf(v, o);
    }
    return v;
}

__device__ float warp_sum(float v, unsigned mask) {
    for(int off = 16; off >0; off /= 2) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

__device__ float block_max(float v) {
    __shared__ float warp_values[32];
    __shared__ float result;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    v = warp_max(v, __activemask());
    if(lane == 0) warp_values[warp] = v;
    __syncthreads();

    if(warp == 0) {
        v = lane < nwarps ? warp_values[lane] : -INFINITY;
        v = warp_max(v, __activemask());
        if(lane == 0) result = v;
    }
    __syncthreads();
    return result;
}

__device__ float block_sum(float v) {
    __shared__ float warp_values[32];
    __shared__ float result;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 32) >> 5;

    v = warp_sum(v, __activemask());
    if (lane == 0) warp_values[warp] = v;
    __syncthreads();

    if(warp == 0) {
        v = lane < nwarps ? warp_values[lane] : 0.0f;
        v = warp_sum(v, __activemask());
        if(lane == 0) result = v;
    }

    __syncthreads();
    return result;
}

__global__ void softmax_block_f32(
    const float* x, float* y,
    int rows, int cols) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        if(row >= rows) return;

        const float* in = x + row * cols;
        float local_max = -INFINITY;
        for(int c = tid; c < cols; c += blockDim.x) {
            local_max = fmaxf(local_max, in[c]);
        }
        float row_max = block_max(local_max);

        float local_sum = 0.0f;
        for (int c = tid; c < cols; c += blockDim.x) {
            local_sum += expf(in[c] - row_max);
        }
        float row_sum = block_sum(local_sum);

        float* out = y + row * cols;
        for (int c = tid; c < cols; c += blockDim.x) {
            float e = expf(in[c] - row_max);
            out[c] = e / row_sum;
        }
    }

void launch_softmax_f32(
    const float* device_input,
    int rows,
    int cols,
    float* device_output,
    int blocks,
    int threads,
    SoftmaxKind kind,
    cudaStream_t stream
) {
     if (device_input == nullptr ||
        device_output == nullptr) {
        throw std::invalid_argument(
            "device pointers must not be null"
        );
    }

    if (rows <= 0 || cols <= 0){
        throw std::invalid_argument(
            "rows and cols must be positive"
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
        case SoftmaxKind::Baseline: {
            const int blocks =
                (rows + threads - 1) / threads;

            softmax_baseline_f32<<<
                blocks, threads, 0, stream
            >>>(
                device_input,
                device_output,
                rows,
                cols
            );
            break;
        }

        case SoftmaxKind::Block:
            softmax_block_f32<<<
                rows, threads, 0, stream
            >>>(
                device_input,
                device_output,
                rows,
                cols
            );
            break;

        default:
            throw std::invalid_argument(
                "unsupported SoftmaxKind"
            );
    }

    cuda_check(
        cudaGetLastError(),
        "launch reduction kernel"
    );
}

const char* softmax_kind_name(SoftmaxKind kind) {
    switch (kind) {
        case SoftmaxKind::Baseline:
            return "baseline";
        case SoftmaxKind::Block:
            return "block";
    }
    return "unknown";
}
