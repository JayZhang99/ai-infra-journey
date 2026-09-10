#include "softmax.cuh"

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

std::vector<float> cpu_softmax(
    const std::vector<float>& input,
    int rows,
    int cols
) {
    std::vector<float> output(input.size());

    for (int row = 0; row < rows; ++row) {
        const int base = row * cols;
        float max_value = -INFINITY;

        for (int col = 0; col < cols; ++col) {
            max_value = std::max(
                max_value,
                input[base + col]
            );
        }

        double sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            const double value = std::exp(
                static_cast<double>(
                    input[base + col] - max_value
                )
            );

            output[base + col] =
                static_cast<float>(value);
            sum += value;
        }

        for (int col = 0; col < cols; ++col) {
            output[base + col] /=
                static_cast<float>(sum);
        }
    }

    return output;
}

void run_case(
    int rows,
    int cols,
    SoftmaxKind kind
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;
    const std::size_t bytes =
        count * sizeof(float);

    std::vector<float> input(count);

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

    const std::vector<float> expected =
        cpu_softmax(input, rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_input),
            bytes
        ),
        "cudaMalloc input"
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

    launch_softmax_f32(
        device_input,
        device_output,
        rows,
        cols,
        256,
        kind
    );

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

    constexpr float atol = 1e-6f;
    constexpr float rtol = 1e-5f;

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

    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            row_sum += actual[row * cols + col];
        }

        if (std::abs(row_sum - 1.0) > 1e-5) {
            throw std::runtime_error(
                "row sum mismatch at row " +
                std::to_string(row)
            );
        }
    }

    cudaFree(device_output);
    cudaFree(device_input);

    std::cout
        << "PASSED"
        << " kind=" << softmax_kind_name(kind)
        << " rows=" << rows
        << " cols=" << cols
        << '\n';
}

}  // namespace

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
            {2, 33},
            {3, 128},
            {4, 1000},
            {2, 2048},
        };

        const SoftmaxKind kinds[] = {
            SoftmaxKind::Baseline,
            SoftmaxKind::Block,
        };

        for (SoftmaxKind kind : kinds) {
            for (const auto& [rows, cols] : cases) {
                run_case(rows, cols, kind);
            }
        }

        bool rejected_null = false;

        try {
            launch_softmax_f32(
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