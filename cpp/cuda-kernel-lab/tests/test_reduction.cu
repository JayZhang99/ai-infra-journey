#include "reduction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

}  // namespace


int main() {
    try {
        int device_count = 0;

        cuda_check(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount"
        );

        if (device_count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 0;
        }

        const std::vector<std::size_t> sizes = {
            1,
            31,
            32,
            33,
            1000,
            1024 * 1024 + 3,
        };

        const ReduceKind kinds[] = {
            ReduceKind::Shared,
            ReduceKind::Warp,
        };

        const std::size_t max_n =
            *std::max_element(
                sizes.begin(),
                sizes.end()
            );

        // 使用全 1 输入，期望结果就是 n。
        std::vector<float> host_input(max_n, 1.0f);

        float* device_input = nullptr;
        float* device_output = nullptr;

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input),
                max_n * sizeof(float)
            ),
            "cudaMalloc input"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&device_output),
                sizeof(float)
            ),
            "cudaMalloc output"
        );

        cuda_check(
            cudaMemcpy(
                device_input,
                host_input.data(),
                max_n * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "copy input"
        );

        constexpr int threads = 256;
        int passed = 0;

        for (ReduceKind kind : kinds) {
            for (std::size_t n : sizes) {
                const std::size_t needed_blocks =
                    (n + threads - 1) / threads;

                const int blocks = static_cast<int>(
                    std::max<std::size_t>(
                        1,
                        std::min<std::size_t>(
                            needed_blocks,
                            128
                        )
                    )
                );

                cuda_check(
                    cudaMemset(
                        device_output,
                        0,
                        sizeof(float)
                    ),
                    "reset output"
                );

                launch_reduction(
                    device_input,
                    n,
                    device_output,
                    blocks,
                    threads,
                    kind
                );

                cuda_check(
                    cudaDeviceSynchronize(),
                    "synchronize kernel"
                );

                float actual = 0.0f;

                cuda_check(
                    cudaMemcpy(
                        &actual,
                        device_output,
                        sizeof(float),
                        cudaMemcpyDeviceToHost
                    ),
                    "copy output"
                );

                const float expected =
                    static_cast<float>(n);

                const float error =
                    std::abs(actual - expected);

                if (error > 1e-4f) {
                    std::cerr
                        << "FAILED"
                        << " kind="
                        << reduce_kind_name(kind)
                        << " n=" << n
                        << " expected=" << expected
                        << " actual=" << actual
                        << " error=" << error
                        << '\n';

                    cudaFree(device_output);
                    cudaFree(device_input);
                    return 1;
                }

                std::cout
                    << "PASSED"
                    << " kind="
                    << reduce_kind_name(kind)
                    << " n=" << n
                    << " result=" << actual
                    << '\n';

                ++passed;
            }
        }

        cudaFree(device_output);
        cudaFree(device_input);

        std::cout
            << "PASSED: "
            << passed
            << " checks\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}