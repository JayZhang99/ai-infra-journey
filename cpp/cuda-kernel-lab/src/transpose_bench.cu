#include "transpose.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct BenchmarkResult {
    int height;
    int width;
    std::string kind;
    std::vector<float> samples_ms;
    float median_ms;
    float p95_ms;
    double effective_gbps;
};

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

const char* benchmark_kind_name(
    const std::optional<TransposeKind>& kind
) {
    return kind.has_value()
        ? transpose_kind_name(*kind)
        : "copy";
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());

    const std::size_t middle =
        values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return 0.5f * (
        values[middle - 1] +
        values[middle]
    );
}

float percentile(
    std::vector<float> values,
    double p
) {
    std::sort(values.begin(), values.end());

    const std::size_t index =
        static_cast<std::size_t>(
            std::ceil(p * values.size())
        ) - 1;

    return values[
        std::min(
            index,
            values.size() - 1
        )
    ];
}

void run_operation(
    const std::optional<TransposeKind>& kind,
    const float* input,
    float* output,
    int height,
    int width,
    std::size_t bytes,
    cudaStream_t stream
) {
    if (!kind.has_value()) {
        cuda_check(
            cudaMemcpyAsync(
                output,
                input,
                bytes,
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "device copy"
        );

        return;
    }

    launch_transpose_f32(
        input,
        output,
        height,
        width,
        256,
        *kind,
        stream
    );
}

std::vector<float> make_reference(
    const std::vector<float>& input,
    int height,
    int width,
    const std::optional<TransposeKind>& kind
) {
    if (!kind.has_value()) {
        return input;
    }

    std::vector<float> output(input.size());

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            output[col * height + row] =
                input[row * width + col];
        }
    }

    return output;
}

BenchmarkResult run_benchmark(
    int height,
    int width,
    const std::optional<TransposeKind>& kind,
    int warmup_batches,
    int repeats,
    int launches_per_sample
) {
    const std::size_t count =
        static_cast<std::size_t>(height) *
        width;

    const std::size_t bytes =
        count * sizeof(float);

    std::vector<float> input(count);

    for (std::size_t i = 0; i < count; ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 257) -
                128
            ) * 0.125f;
    }

    const std::vector<float> expected =
        make_reference(
            input,
            height,
            width,
            kind
        );

    std::vector<float> actual(
        count,
        std::numeric_limits<float>::
            quiet_NaN()
    );

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_input
            ),
            bytes
        ),
        "allocate input"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_output
            ),
            bytes
        ),
        "allocate output"
    );

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

    cuda_check(
        cudaMemcpyAsync(
            device_input,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice,
            stream
        ),
        "copy benchmark input"
    );

    cuda_check(
        cudaMemcpyAsync(
            device_output,
            actual.data(),
            bytes,
            cudaMemcpyHostToDevice,
            stream
        ),
        "initialize benchmark output"
    );

    // Warmup 不计时。
    for (
        int batch = 0;
        batch < warmup_batches;
        ++batch
    ) {
        for (
            int launch = 0;
            launch < launches_per_sample;
            ++launch
        ) {
            run_operation(
                kind,
                device_input,
                device_output,
                height,
                width,
                bytes,
                stream
            );
        }
    }

    cuda_check(
        cudaStreamSynchronize(stream),
        "warmup synchronize"
    );

    std::vector<float> samples;
    samples.reserve(repeats);

    for (
        int repeat = 0;
        repeat < repeats;
        ++repeat
    ) {
        cuda_check(
            cudaEventRecord(start, stream),
            "record start"
        );

        for (
            int launch = 0;
            launch < launches_per_sample;
            ++launch
        ) {
            run_operation(
                kind,
                device_input,
                device_output,
                height,
                width,
                bytes,
                stream
            );
        }

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
            "get elapsed time"
        );

        samples.push_back(
            elapsed_ms /
            launches_per_sample
        );
    }

    // 正确性检查放在计时区间之外。
    cuda_check(
        cudaMemcpyAsync(
            actual.data(),
            device_output,
            bytes,
            cudaMemcpyDeviceToHost,
            stream
        ),
        "copy benchmark output"
    );

    cuda_check(
        cudaStreamSynchronize(stream),
        "correctness synchronize"
    );

    for (std::size_t i = 0; i < count; ++i) {
        if (
            !std::isfinite(actual[i]) ||
            actual[i] != expected[i]
        ) {
            throw std::runtime_error(
                "correctness failure"
                " kind=" +
                std::string(
                    benchmark_kind_name(kind)
                ) +
                " height=" +
                std::to_string(height) +
                " width=" +
                std::to_string(width) +
                " index=" +
                std::to_string(i)
            );
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);

    const float median_ms =
        median(samples);

    const float p95_ms =
        percentile(samples, 0.95);

    // 一次 Transpose 或 Copy：读一遍、写一遍。
    const double logical_bytes =
        2.0 * static_cast<double>(bytes);

    const double effective_gbps =
        logical_bytes /
        (
            static_cast<double>(median_ms) *
            1.0e6
        );

    return {
        height,
        width,
        benchmark_kind_name(kind),
        samples,
        median_ms,
        p95_ms,
        effective_gbps,
    };
}


}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::invalid_argument(
                "usage: transpose_bench "
                "<output.json> "
                "<run_id> "
                "<git_revision> "
                "<nvidia_driver_version>"
            );
        }

        const std::string output_path =
            argv[1];

        const std::string run_id =
            argv[2];

        const std::string git_revision =
            argv[3];

        const std::string driver_version =
            argv[4];

        constexpr int warmup_batches = 5;
        constexpr int repeats = 50;
        constexpr int launches_per_sample = 100;

        const std::vector<
            std::pair<int, int>
        > shapes = {
            {256, 256},
            {1000, 1024},
            {1024, 1024},
            {4096, 4096},
        };

        const std::optional<TransposeKind> kinds[] = {
            std::nullopt,
            TransposeKind::Naive,
            TransposeKind::Tiled,
            TransposeKind::Padding,
        };

        std::vector<BenchmarkResult> results;

        for (
            const auto& [height, width] :
            shapes
        ) {
            for (const auto& kind : kinds) {
                results.push_back(
                    run_benchmark(
                        height,
                        width,
                        kind,
                        warmup_batches,
                        repeats,
                        launches_per_sample
                    )
                );
            }
        }

        const std::filesystem::path path(
            output_path
        );

        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(
                path.parent_path()
            );
        }

        std::ofstream output(path);

        if (!output.is_open()) {
            throw std::runtime_error(
                "failed to open output file"
            );
        }

        output << std::setprecision(9);

        output
            << "{\n"
            << "  \"schema_version\": 2,\n"
            << "  \"benchmark\": "
            << std::quoted("transpose_f32")
            << ",\n"
            << "  \"run_id\": "
            << std::quoted(run_id)
            << ",\n"
            << "  \"git_revision\": "
            << std::quoted(git_revision)
            << ",\n"
            << "  \"nvidia_driver_version\": "
            << std::quoted(driver_version)
            << ",\n"
            << "  \"storage_dtype\": \"float32\",\n"
            << "  \"accumulator_dtype\": null,\n"
            << "  \"measurement\": "
            << std::quoted(
                "batched_average_per_operation"
            )
            << ",\n"
            << "  \"timer\": "
            << std::quoted(
                "cuda_event_same_stream"
            )
            << ",\n"
            << "  \"time_unit\": \"ms\",\n"
            << "  \"logical_bytes_rule\": "
            << std::quoted(
                "2*height*width*sizeof(float)"
            )
            << ",\n"
            << "  \"warmup_batches\": "
            << warmup_batches
            << ",\n"
            << "  \"repeats\": "
            << repeats
            << ",\n"
            << "  \"launches_per_sample\": "
            << launches_per_sample
            << ",\n"
            << "  \"block\": [32, 8, 1],\n"
            << "  \"results\": [\n";

        for (
            std::size_t i = 0;
            i < results.size();
            ++i
        ) {
            const BenchmarkResult& result =
                results[i];

            output
                << "    {\n"
                << "      \"height\": "
                << result.height
                << ",\n"
                << "      \"width\": "
                << result.width
                << ",\n"
                << "      \"kind\": "
                << std::quoted(result.kind)
                << ",\n"
                << "      \"median_ms\": "
                << result.median_ms
                << ",\n"
                << "      \"p95_batch_mean_ms\": "
                << result.p95_ms
                << ",\n"
                << "      \"effective_gbps\": "
                << result.effective_gbps
                << ",\n"
                << "      \"samples_ms\": [";

            for (
                std::size_t j = 0;
                j < result.samples_ms.size();
                ++j
            ) {
                if (j != 0) {
                    output << ", ";
                }

                output
                    << result.samples_ms[j];
            }

            output << "]\n    }";

            if (i + 1 != results.size()) {
                output << ',';
            }

            output << '\n';
        }

        output
            << "  ]\n"
            << "}\n";

        output.flush();

        if (!output.good()) {
            throw std::runtime_error(
                "failed while writing JSON"
            );
        }

        std::cout
            << "WROTE: "
            << output_path
            << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "FAILED: "
            << error.what()
            << '\n';

        return 1;
    }
}
