#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class ReduceKind{
    Shared,
    Warp,
};

const char* reduce_kind_name(ReduceKind kind);

void launch_reduction(
    const float* device_input,
    std::size_t n,
    float* device_output,
    int blocks,
    int threads,
    ReduceKind kind,
    cudaStream_t stream = nullptr
);