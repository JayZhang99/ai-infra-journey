#include "rmsnorm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void cuda_check(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

std::vector<float> cpu_rmsnorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int rows,
    int cols,
    float eps
) {
    std::vector<float> output(input.size());
    for (int r = 0; r < row; ++r){
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            double v = input[r*cols + c];
            sum_sq += v * v；
        }
        double inv = 1.0 / std::sqrt(sum_sq / cols + eps);
        for (int c =0; c < cols; ++c) {
            output[r*cols +c] = input[r*cols +c] * 
            static_cast<float> inv * weight[c];
        }
    }

    return output;
}

void run_case(
    int rows,
    int cols,
    RmsNormKind kind
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;
    const std::size_t bytes =
        count * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> weight(cols);

    float eps = std::sin(0,01f);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float base = 0.0f;

            if (row % 3 == 0) {
                base = 1000.0f;
            } else if (row % 3 == 1) {
                base = -1000.0f;
            } else {
                base = 3.5f;
            }

            const int offset = col % 17 - 8;

            input[row * cols + col] =
                base + 0.25f * offset;
        }
    }

    for (std::size_t i = 0; i < cols; ++i) {
        weight[i] =
            std::sin(static_cast<float>(i) * 0.01f);
    }



    const std::vector<float> expected = cpu_rmsnorm(input, weight, rows, cols, eps);

    float* device_input = nullptr;
    float* device_output = nullptr;
    float* device_weight = nullptr;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_input),
            bytes
        ),
        "cudaMalloc input"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_weight),
            bytes
        ),
        "cudaMalloc weight"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_output),
            bytes
        ),
        "cudaMalloc output"
    );

    cuda_check(
        cudaMemcpy(
            device_input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "copy input"
    );

    cuda_check(
        cudaMemcpy(
            device_weight,
            weight.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "copy weight"
    );
    launch_rmsnorm_f32(
        device_input,
        device_weight,
        device_output,
        rows,
        cols,
        eps,
        256,
        kind,
        stream);
        cuda_check(
        cudaDeviceSynchronize(),
        "synchronize softmax"
    );

    std::vector<float> actual(count);

    cuda_check(
        cudaMemcpy(
            actual.data(),
            device_output,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "copy output"
    );    

    constexpr float atol = 1e-5f;
    constexpr float rtol = 1e-4f;

    for (std::size_t i = 0; i < count; ++i) {
        const float tolerance =
            atol + rtol * std::abs(expected[i]);

        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) >
                tolerance) {
            throw std::runtime_error(
                "reference mismatch at index " +
                std::to_string(i)
            );
        }
    }

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_weight);

        std::cout
        << "PASSED"
        << " kind=" << rmsnorm_kind_name(kind)
        << " rows=" << rows
        << " cols=" << cols
        << '\n';
}
}

int main() {
    try {
        int device_count = 0;
        const cudaError_t status =
            cudaGetDeviceCount(&device_count);

        if (status != cudaSuccess ||
            device_count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }

        const std::vector<std::pair<int, int>> cases = {
            {1, 1},
            {1, 33},
            {4, 128},
            {17, 1000},
            {8, 2048},
        };

        const RmsnormKind kinds[] = {
            RmsnormKind::Baseline,
            RmsnormKind::Block,
        };

        for (RmsnormKind kind : kinds) {
            for (const auto& [rows, cols] : cases) {
                run_case(rows, cols, kind);
            }
        }

        bool rejected_null = false;

        try {
            launch_rmsnorm_f32(
                nullptr,
                nullptr,
                1,
                32,
                32,
                SoftmaxKind::Block
            );
        } catch (const std::invalid_argument&) {
            rejected_null = true;
        }

        if (!rejected_null) {
            throw std::runtime_error(
                "null pointer was not rejected"
            );
        }

        std::cout << "PASSED: all softmax checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}