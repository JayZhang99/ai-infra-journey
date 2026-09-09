#include "reduction.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>


namespace {

constexpr unsigned kFullMask = 0xffffffffu;


void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}


__device__ __forceinline__
float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2;
         offset > 0;
         offset /= 2) {
        value += __shfl_down_sync(
            kFullMask,
            value,
            offset
        );
    }

    return value;
}


__global__
void shared_reduce_kernel(
    const float* input,
    std::size_t n,
    float* output
) {
    extern __shared__ float scratch[];

    const unsigned tid = threadIdx.x;

    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + tid;

    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) *
        blockDim.x;

    float thread_sum = 0.0f;

    for (; index < n; index += grid_stride) {
        thread_sum += input[index];
    }

    scratch[tid] = thread_sum;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, scratch[0]);
    }
}


__global__
void warp_reduce_kernel(
    const float* input,
    std::size_t n,
    float* output
) {
    __shared__ float warp_sums[32];

    const unsigned tid = threadIdx.x;
    const unsigned lane_id = tid % warpSize;
    const unsigned warp_id = tid / warpSize;
    const unsigned warp_count = blockDim.x / warpSize;

    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + tid;

    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) *
        blockDim.x;

    float thread_sum = 0.0f;

    for (; index < n; index += grid_stride) {
        thread_sum += input[index];
    }

    // 第一层：每个 Warp 内部归约。
    const float local_warp_sum =
        warp_reduce_sum(thread_sum);

    // 每个 Warp 由 lane 0 写出一个结果。
    if (lane_id == 0) {
        warp_sums[warp_id] = local_warp_sum;
    }

    // 等待所有 Warp 完成写入。
    __syncthreads();

    // 第二层：第一个 Warp 汇总所有 Warp 的结果。
    if (warp_id == 0) {
        float block_sum =
            lane_id < warp_count
                ? warp_sums[lane_id]
                : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

}  // namespace


const char* reduce_kind_name(ReduceKind kind) {
    switch (kind) {
        case ReduceKind::Shared:
            return "shared";

        case ReduceKind::Warp:
            return "warp";
    }

    return "unknown";
}


void launch_reduction(
    const float* device_input,
    std::size_t n,
    float* device_output,
    int blocks,
    int threads,
    ReduceKind kind,
    cudaStream_t stream
) {
    if (device_input == nullptr ||
        device_output == nullptr) {
        throw std::invalid_argument(
            "device pointers must not be null"
        );
    }

    if (n == 0) {
        throw std::invalid_argument(
            "n must be positive"
        );
    }

    if (blocks <= 0) {
        throw std::invalid_argument(
            "blocks must be positive"
        );
    }

    const bool is_power_of_two =
        threads > 0 &&
        (threads & (threads - 1)) == 0;

    if (!is_power_of_two ||
        threads < 32 ||
        threads > 1024) {
        throw std::invalid_argument(
            "threads must be a power of two "
            "between 32 and 1024"
        );
    }

    switch (kind) {
        case ReduceKind::Shared: {
            const std::size_t shared_bytes =
                static_cast<std::size_t>(threads) *
                sizeof(float);

            shared_reduce_kernel<<<
                blocks,
                threads,
                shared_bytes,
                stream
            >>>(
                device_input,
                n,
                device_output
            );

            break;
        }

        case ReduceKind::Warp:
            warp_reduce_kernel<<<
                blocks,
                threads,
                0,
                stream
            >>>(
                device_input,
                n,
                device_output
            );

            break;
    }

    cuda_check(
        cudaGetLastError(),
        "launch reduction kernel"
    );
}