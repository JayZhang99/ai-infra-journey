#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class SoftmaxKind {
    Baseline,
    Block,
};
const char* softmax_kind_name(SoftmaxKind kind);

void launch_softmax_f32(
    const float* input,
    float* output,
    int rows,
    int cols,
    int threads,
    SoftmaxKind kind,
    cudaStream_t stream = nullptr
);
