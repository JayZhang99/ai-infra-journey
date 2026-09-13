#include "transpose.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cmath>

namespace {

constexpr int kTile = 32;
constexpr int kBlockRows = 8;
constexpr int kThreads = kTile * kBlockRows;

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}
}

const char* transpose_kind_name(TransposeKind kind) {
    switch (kind) {
        case TransposeKind::Naive:
            return "naive";

        case TransposeKind::Tiled:
            return "tiled";

        case TransposeKind::Padding:
            return "padding";
    }

    return "unknown";
}

__global__ void transpose_naive_f32(
    const float* input, float* output, int height, int width
) {
    const int input_col =
        blockIdx.x * kTile +
        threadIdx.x;

    const int input_row =
        blockIdx.y * kTile +
        threadIdx.y;

    for (
        int j = 0;
        j < kTile;
        j += kBlockRows
    ) {
        const int row = input_row + j;

        if (
            row < height &&
            input_col < width
        ) {
            output[
                input_col * height + row
            ] = input[
                row * width + input_col
            ];
        }
    }

}

template <int Padding>
__global__ void transpose_tiled_f32(
    const float* input,
    float* output,
    int height,
    int width
) {
    __shared__ float tile[
        kTile
    ][
        kTile + Padding
    ];

    const int input_col =
        blockIdx.x * kTile +
        threadIdx.x;

    const int input_row =
        blockIdx.y * kTile +
        threadIdx.y;

    for (
        int j = 0;
        j < kTile;
        j += kBlockRows
    ) {
        if (
            input_col < width &&
            input_row + j < height
        ) {
            tile[
                threadIdx.y + j
            ][
                threadIdx.x
            ] = input[
                (input_row + j) * width +
                input_col
            ];
        }
    }

    __syncthreads();

    const int output_col =
        blockIdx.y * kTile +
        threadIdx.x;

    const int output_row =
        blockIdx.x * kTile +
        threadIdx.y;

    for (
        int j = 0;
        j < kTile;
        j += kBlockRows
    ) {
        if (
            output_col < height &&
            output_row + j < width
        ) {
            output[
                (output_row + j) * height +
                output_col
            ] = tile[
                threadIdx.x
            ][
                threadIdx.y + j
            ];
        }
    }
}


void launch_transpose_f32(
    const float* input,
    float* output,
    int height,
    int width,
    int threads,
    TransposeKind kind,
    cudaStream_t stream
) {
    if (input == nullptr ||
        output == nullptr) {
        throw std::invalid_argument(
            "device pointers must not be null"
        );
    }

    if (height <= 0 || width <=0) {
        throw std::invalid_argument(
            "heights and weights must be positvie"
        );
    }

    if (threads != kThreads) {
        throw std::invalid_argument(
            "transpose requires exactly "
            "256 threads arranged as 32x8"
        );
    }

    const dim3 block(
        kTile,
        kBlockRows
    );

    const dim3 grid(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile
    );

switch (kind) {
        case TransposeKind::Naive:
            transpose_naive_f32<<<
                grid,
                block,
                0,
                stream
            >>>(
                input,
                output,
                height,
                width
            );
            break;

        case TransposeKind::Tiled:
            transpose_tiled_f32<0><<<
                grid,
                block,
                0,
                stream
            >>>(
                input,
                output,
                height,
                width
            );
            break;

        case TransposeKind::Padding:
            transpose_tiled_f32<1><<<
                grid,
                block,
                0,
                stream
            >>>(
                input,
                output,
                height,
                width
            );
            break;

        default:
            throw std::invalid_argument(
                "unsupported TransposeKind"
            );
    }

    cuda_check(
        cudaGetLastError(),
        "launch transpose kernel"
    );

}