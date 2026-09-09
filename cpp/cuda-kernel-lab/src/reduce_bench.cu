#include "reduction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


namespace {

struct Args {
    std::size_t n = 1024 * 1024;
    int threads = 256;
    int warmup = 10;
    int repeats = 100;

    std::filesystem::path output =
        "benchmarks/cuda/reduction.json";
};


struct Result {
    std::string kind;

    float gpu_sum = 0.0f;
    double cpu_sum = 0.0;
    double absolute_error = 0.0;

    float median_ms = 0.0f;
    float p95_ms = 0.0f;

    std::vector<float> samples_ms;
};


void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}


std::string next_value(
    int& index,
    int argc,
    char** argv
) {
    ++index;

    if (index >= argc) {
        throw std::invalid_argument(
            "missing CLI argument value"
        );
    }

    return argv[index];
}


Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (key == "--n") {
            args.n = std::stoull(
                next_value(i, argc, argv)
            );
        } else if (key == "--threads") {
            args.threads = std::stoi(
                next_value(i, argc, argv)
            );
        } else if (key == "--warmup") {
            args.warmup = std::stoi(
                next_value(i, argc, argv)
            );
        } else if (key == "--repeats") {
            args.repeats = std::stoi(
                next_value(i, argc, argv)
            );
        } else if (key == "--output") {
            args.output = next_value(
                i,
                argc,
                argv
            );
        } else {
            throw std::invalid_argument(
                "unknown argument: " + key
            );
        }
    }

    if (args.n == 0) {
        throw std::invalid_argument(
            "--n must be positive"
        );
    }

    if (args.warmup < 0) {
        throw std::invalid_argument(
            "--warmup must not be negative"
        );
    }

    if (args.repeats <= 0) {
        throw std::invalid_argument(
            "--repeats must be positive"
        );
    }

    return args;
}


float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return 0.5f *
        (values[middle - 1] + values[middle]);
}


float percentile_nearest(
    std::vector<float> values,
    double percentile
) {
    std::sort(values.begin(), values.end());

    const std::size_t rank =
        static_cast<std::size_t>(
            std::ceil(
                percentile *
                static_cast<double>(values.size())
            )
        );

    const std::size_t index =
        std::max<std::size_t>(rank, 1) - 1;

    return values[
        std::min(index, values.size() - 1)
    ];
}


double cpu_reference(
    const std::vector<float>& input
) {
    double total = 0.0;

    for (float value : input) {
        total += value;
    }

    return total;
}


Result run_case(
    ReduceKind kind,
    const Args& args,
    const std::vector<float>& host_input,
    const float* device_input,
    float* device_output,
    int blocks
) {
    Result result;
    result.kind = reduce_kind_name(kind);
    result.cpu_sum = cpu_reference(host_input);

    // 正确性运行。
    cuda_check(
        cudaMemset(
            device_output,
            0,
            sizeof(float)
        ),
        "reset correctness output"
    );

    launch_reduction(
        device_input,
        args.n,
        device_output,
        blocks,
        args.threads,
        kind
    );

    cuda_check(
        cudaDeviceSynchronize(),
        "correctness synchronize"
    );

    cuda_check(
        cudaMemcpy(
            &result.gpu_sum,
            device_output,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy correctness output"
    );

    result.absolute_error = std::abs(
        static_cast<double>(result.gpu_sum) -
        result.cpu_sum
    );

    if (result.absolute_error > 1e-4) {
        throw std::runtime_error(
            result.kind +
            " result does not match CPU reference"
        );
    }

    // Warmup 不进入样本。
    for (int i = 0; i < args.warmup; ++i) {
        cuda_check(
            cudaMemsetAsync(
                device_output,
                0,
                sizeof(float)
            ),
            "warmup reset"
        );

        launch_reduction(
            device_input,
            args.n,
            device_output,
            blocks,
            args.threads,
            kind
        );
    }

    cuda_check(
        cudaDeviceSynchronize(),
        "warmup synchronize"
    );

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(
        cudaEventCreate(&start),
        "create start event"
    );

    cuda_check(
        cudaEventCreate(&stop),
        "create stop event"
    );

    result.samples_ms.reserve(args.repeats);

    for (int repeat = 0;
         repeat < args.repeats;
         ++repeat) {
        // Reset 排除在 Kernel 计时之外。
        cuda_check(
            cudaMemsetAsync(
                device_output,
                0,
                sizeof(float)
            ),
            "sample reset"
        );

        cuda_check(
            cudaEventRecord(start),
            "record start"
        );

        launch_reduction(
            device_input,
            args.n,
            device_output,
            blocks,
            args.threads,
            kind
        );

        cuda_check(
            cudaEventRecord(stop),
            "record stop"
        );

        cuda_check(
            cudaEventSynchronize(stop),
            "wait stop"
        );

        float elapsed_ms = 0.0f;

        cuda_check(
            cudaEventElapsedTime(
                &elapsed_ms,
                start,
                stop
            ),
            "read elapsed time"
        );

        result.samples_ms.push_back(elapsed_ms);
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    result.median_ms = median(result.samples_ms);
    result.p95_ms = percentile_nearest(
        result.samples_ms,
        0.95
    );

    return result;
}


void write_json(
    const Args& args,
    int blocks,
    const std::vector<Result>& results
) {
    const auto parent = args.output.parent_path();

    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(args.output);

    if (!output) {
        throw std::runtime_error(
            "cannot open output file"
        );
    }

    output << std::fixed << std::setprecision(8);

    output << "{\n";
    output << "  \"n\": " << args.n << ",\n";
    output << "  \"threads\": " << args.threads << ",\n";
    output << "  \"blocks\": " << blocks << ",\n";
    output << "  \"warmup\": " << args.warmup << ",\n";
    output << "  \"repeats\": " << args.repeats << ",\n";
    output << "  \"results\": [\n";

    for (std::size_t i = 0;
         i < results.size();
         ++i) {
        const Result& result = results[i];

        output << "    {\n";
        output << "      \"kind\": \""
               << result.kind << "\",\n";
        output << "      \"cpu_sum\": "
               << result.cpu_sum << ",\n";
        output << "      \"gpu_sum\": "
               << result.gpu_sum << ",\n";
        output << "      \"absolute_error\": "
               << result.absolute_error << ",\n";
        output << "      \"median_ms\": "
               << result.median_ms << ",\n";
        output << "      \"p95_ms\": "
               << result.p95_ms << ",\n";
        output << "      \"samples_ms\": [";

        for (std::size_t j = 0;
             j < result.samples_ms.size();
             ++j) {
            if (j > 0) {
                output << ", ";
            }

            output << result.samples_ms[j];
        }

        output << "]\n";
        output << "    }";

        if (i + 1 != results.size()) {
            output << ",";
        }

        output << "\n";
    }

    output << "  ]\n";
    output << "}\n";
}

}  // namespace


int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        int device = 0;
        cuda_check(
            cudaGetDevice(&device),
            "get CUDA device"
        );

        cudaDeviceProp properties{};
        cuda_check(
            cudaGetDeviceProperties(
                &properties,
                device
            ),
            "get CUDA properties"
        );

        const std::size_t needed_blocks =
            (args.n + args.threads - 1) /
            args.threads;

        const std::size_t block_limit =
            static_cast<std::size_t>(
                properties.multiProcessorCount
            ) * 8;

        const int blocks = static_cast<int>(
            std::max<std::size_t>(
                1,
                std::min(
                    needed_blocks,
                    block_limit
                )
            )
        );

        std::vector<float> host_input(args.n);

        for (std::size_t i = 0; i < args.n; ++i) {
            host_input[i] =
                0.25f *
                static_cast<float>((i % 4) + 1);
        }

        float* device_input = nullptr;
        float* device_output = nullptr;

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input),
                args.n * sizeof(float)
            ),
            "allocate input"
        );

        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&device_output),
                sizeof(float)
            ),
            "allocate output"
        );

        cuda_check(
            cudaMemcpy(
                device_input,
                host_input.data(),
                args.n * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "copy input"
        );

        std::vector<Result> results;

        results.push_back(
            run_case(
                ReduceKind::Shared,
                args,
                host_input,
                device_input,
                device_output,
                blocks
            )
        );

        results.push_back(
            run_case(
                ReduceKind::Warp,
                args,
                host_input,
                device_input,
                device_output,
                blocks
            )
        );

        write_json(args, blocks, results);

        for (const Result& result : results) {
            std::cout
                << result.kind
                << " median=" << result.median_ms
                << " ms"
                << " p95=" << result.p95_ms
                << " ms"
                << " error=" << result.absolute_error
                << '\n';
        }

        cudaFree(device_output);
        cudaFree(device_input);

        std::cout
            << "saved: "
            << args.output
            << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}