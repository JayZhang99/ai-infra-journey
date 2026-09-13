#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class RmsNormKind {
    Baseline,
    Block,
};

const char* rmsnorm_kind_name(RmsNormKind kind);

void launch_rmsnorm_f32(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps,
    int threads,
    RmsNormKind kind,
    cudaStream_t stream = nullptr
);