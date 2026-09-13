#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class TransposeKind {
    Naive,
    Tiled,
    Padding
};

const char* transpose_kind_name(TransposeKind kind);

void launch_transpose_f32(
    const float* input,
    float* output,
    int h,
    int w,
    int threads,
    TransposeKind kind,
    cudaStream_t stream = nullptr
);