#include "softmax.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct BenchmarkResult {
    int rows;
    int cols;
    int threads;
    SoftmaxKind kind;
    std::vector<float> samples_ms;
    float median_ms;
    float p95_ms;
};

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

float percentile(
    std::vector<float> values,
    double percentile_value
) {
    std::sort(values.begin(), values.end());

    const std::size_t index =
        static_cast<std::size_t>(
            std::ceil(
                percentile_value * values.size()
            )
        ) - 1;

    return values[
        std::min(index, values.size() - 1)
    ];
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());

    const std::size_t middle =
        values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return 0.5f * (
        values[middle - 1] + values[middle]
    );
}

BenchmarkResult run_benchmark(
    int rows,
    int cols,
    int threads,
    SoftmaxKind kind,
    int warmup,
    int repeats
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;
    const std::size_t bytes =
        count * sizeof(float);

    std::vector<float> input(count);

    for (std::size_t i = 0; i < count; ++i) {
        input[i] =
            std::sin(static_cast<float>(i) * 0.01f);
    }

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

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(
        cudaStreamCreate(&stream),
        "create stream"
    );
    cuda_check(
        cudaEventCreate(&start),
        "create start event"
    );
    cuda_check(
        cudaEventCreate(&stop),
        "create stop event"
    );

    // Warmup，不记录。
    for (int i = 0; i < warmup; ++i) {
        launch_softmax_f32(
            device_input,
            device_output,
            rows,
            cols,
            threads,
            kind,
            stream
        );
    }

    cuda_check(
        cudaStreamSynchronize(stream),
        "warmup synchronize"
    );

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int repeat = 0;
         repeat < repeats;
         ++repeat) {
        cuda_check(
            cudaEventRecord(start, stream),
            "record start"
        );

        launch_softmax_f32(
            device_input,
            device_output,
            rows,
            cols,
            threads,
            kind,
            stream
        );

        cuda_check(
            cudaEventRecord(stop, stream),
            "record stop"
        );

        cuda_check(
            cudaEventSynchronize(stop),
            "synchronize stop"
        );

        float elapsed_ms = 0.0f;

        cuda_check(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop
            ),
            "elapsed time"
        );

        samples.push_back(elapsed_ms);
    }

    // 正确性只在计时外检查一次。
    std::vector<float> output(count);

    cuda_check(
        cudaMemcpy(
            output.data(),
            device_output,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "copy output"
    );

    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            const float value =
                output[row * cols + col];

            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "non-finite benchmark output"
                );
            }

            sum += value;
        }

        if (std::abs(sum - 1.0) > 1e-5) {
            throw std::runtime_error(
                "benchmark row sum mismatch"
            );
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);

    return {
        rows,
        cols,
        threads,
        kind,
        samples,
        median(samples),
        percentile(samples, 0.95),
    };
}

int main(int argc, char** argv) {
    try {
        const std::string output_path =
            argc > 1
                ? argv[1]
                : "softmax_benchmark.json";

        constexpr int threads = 256;
        constexpr int warmup = 10;
        constexpr int repeats = 100;

        const int row_values[] = {
            1, 32, 256
        };

        const int col_values[] = {
            32, 128, 512, 1024, 2048
        };

        const SoftmaxKind kinds[] = {
            SoftmaxKind::Baseline,
            SoftmaxKind::Block,
        };

        std::vector<BenchmarkResult> results;

        for (int rows : row_values) {
            for (int cols : col_values) {
                for (SoftmaxKind kind : kinds) {
                    results.push_back(
                        run_benchmark(
                            rows,
                            cols,
                            threads,
                            kind,
                            warmup,
                            repeats
                        )
                    );
                }
            }
        }

        std::ofstream output(output_path);

        output << "{\n  \"results\": [\n";

        for (std::size_t i = 0;
             i < results.size();
             ++i) {
            const auto& result = results[i];

            output
                << "    {\n"
                << "      \"rows\": "
                << result.rows << ",\n"
                << "      \"cols\": "
                << result.cols << ",\n"
                << "      \"threads\": "
                << result.threads << ",\n"
                << "      \"kind\": "
                << std::quoted(
                    softmax_kind_name(result.kind)
                )
                << ",\n"
                << "      \"median_ms\": "
                << result.median_ms << ",\n"
                << "      \"p95_ms\": "
                << result.p95_ms << ",\n"
                << "      \"samples_ms\": [";

            for (std::size_t sample = 0;
                 sample < result.samples_ms.size();
                 ++sample) {
                if (sample != 0) {
                    output << ", ";
                }

                output
                    << result.samples_ms[sample];
            }

            output << "]\n    }";

            if (i + 1 != results.size()) {
                output << ",";
            }

            output << "\n";
        }

        output << "  ]\n}\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "FAILED: "
            << error.what()
            << '\n';

        return 1;
    }
}