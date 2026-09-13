#include "rmsnorm.cuh"

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
#include <sstream>

struct BenchmarkResult{
    int rows;
    int cols;
    int threads;
    RmsNormKind kind;
    std::vector<float> samples_ms;
    float median_ms;
    float p95_ms;
};

struct BenchmarkArgs {
    std::string output_path;
    std::string run_id;
    std::string git_revision;
    std::string nvidia_driver_version;
};

struct EnvironmentMetadata {
    int device_ordinal;
    std::string gpu_name;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    std::size_t total_global_memory_bytes;
    std::string cuda_compiled_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_api_version;
};

namespace {

BenchmarkArgs parse_args(
    int argc,
    char** argv
) {
    if (argc != 5) {
        throw std::invalid_argument(
            "usage: rmsnorm_bench "
            "<output.json> "
            "<run_id> "
            "<git_revision> "
            "<nvidia_driver_version>"
        );
    }

    BenchmarkArgs args{
        argv[1],
        argv[2],
        argv[3],
        argv[4],
    };

    if (args.output_path.empty()) {
        throw std::invalid_argument(
            "output path must not be empty"
        );
    }

    if (args.run_id.empty()) {
        throw std::invalid_argument(
            "run id must not be empty"
        );
    }

    if (args.git_revision.empty()) {
        throw std::invalid_argument(
            "git revision must not be empty"
        );
    }

    if (args.nvidia_driver_version.empty()) {
        throw std::invalid_argument(
            "NVIDIA driver version must not be empty"
        );
    }

    return args;
}

std::string format_cuda_version(
    int encoded_version
) {
    const int major =
        encoded_version / 1000;

    const int minor =
        (encoded_version % 1000) / 10;

    std::ostringstream stream;
    stream << major << '.' << minor;
    return stream.str();
}




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

EnvironmentMetadata read_environment() {
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
        "get CUDA device properties"
    );

    int runtime_version = 0;

    cuda_check(
        cudaRuntimeGetVersion(
            &runtime_version
        ),
        "get CUDA runtime version"
    );

    int driver_api_version = 0;

    cuda_check(
        cudaDriverGetVersion(
            &driver_api_version
        ),
        "get CUDA driver API version"
    );

    return EnvironmentMetadata{
        device,
        properties.name,
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        properties.maxThreadsPerBlock,
        properties.totalGlobalMem,
        format_cuda_version(CUDART_VERSION),
        format_cuda_version(runtime_version),
        format_cuda_version(driver_api_version),
    };
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

void cpu_reference(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; ++r){
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            double v = input[r*cols + c];
            sum_sq += v * v;
        }
        double inv = 1.0 / std::sqrt(sum_sq / cols + eps);
        for (int c =0; c < cols; ++c) {
            output[r*cols +c] = input[r*cols +c] * 
            static_cast<float> (inv) * weight[c];
        }
    }

}

BenchmarkResult run_benchmark(
    int rows,
    int cols,
    float eps,
    int threads,
    RmsNormKind kind,
    int warmup,
    int repeats,
    int launches_per_sample
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;

    const std::size_t tensor_bytes =
    static_cast<std::size_t>(rows) * cols * sizeof(float);

const std::size_t weight_bytes =
    static_cast<std::size_t>(cols) * sizeof(float);
    
    std::vector<float> input(count);
    std::vector<float> weight(cols);

    for (std::size_t i = 0; i < count; ++i) {
        input[i] =
            std::sin(static_cast<float>(i) * 0.01f);
    }

    for (std::size_t i = 0; i < cols; ++i) {
        weight[i] =
            std::sin(static_cast<float>(i) * 0.01f);
    }


    float* device_input = nullptr;
    float* device_output = nullptr;
    float* device_weight = nullptr;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_input),
            tensor_bytes
        ),
        "cudaMalloc input"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_weight),
            weight_bytes
        ),
        "cudaMalloc weight"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_output),
            tensor_bytes
        ),
        "cudaMalloc output"
    );

    cuda_check(
        cudaMemcpy(
            device_input,
            input.data(),
            tensor_bytes,
            cudaMemcpyHostToDevice
        ),
        "copy input"
    );

    cuda_check(
        cudaMemcpy(
            device_weight,
            weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice
        ),
        "copy weight"
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
        for (int k = 0; k < launches_per_sample; ++k) {
            launch_rmsnorm_f32(
                device_input,
                device_weight,
                device_output,
                rows,
                cols,
                eps,
                threads,
                kind,
                stream);
        }
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

        for (int k = 0;
            k < launches_per_sample;
            ++k) {
                launch_rmsnorm_f32(
                    device_input,
                    device_weight,
                    device_output,
                    rows,
                    cols,
                    eps,
                    threads,
                    kind,
                    stream
                );}
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
        const float per_launch_ms = elapsed_ms / launches_per_sample;
        samples.push_back(per_launch_ms);
    }

    // 正确性只在计时外检查一次。
    std::vector<float> output(count);

    cuda_check(
        cudaMemcpy(
            output.data(),
            device_output,
            tensor_bytes,
            cudaMemcpyDeviceToHost
        ),
        "copy output"
    );

    std::vector<float> reference(count);
    cpu_reference(input.data(), weight.data(), reference.data(), rows, cols, eps);

    for(size_t i = 0; i < count; ++i) {
        float expected = reference[i];
        float actual = output[i];
        float atol = 1e-5;
        float rtol = 1e-4;
        float tolerance = atol + rtol * std::abs(expected);
        
        if(!std::isfinite(actual) || 
        std::abs(actual - expected) > tolerance) {
            throw std::runtime_error(
                "rmsnorm mismatch");
        }
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_weight);

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
}

int main(int argc, char** argv) {
    try {

        const BenchmarkArgs args =
            parse_args(argc, argv);

        const EnvironmentMetadata environment =
            read_environment();

        const std::filesystem::path output_path(
            args.output_path
        );

        const std::filesystem::path parent =
            output_path.parent_path();

        if (!parent.empty()) {
            std::filesystem::create_directories(
                parent
            );
        }

        std::ofstream output(output_path);

        if (!output.is_open()) {
            throw std::runtime_error(
                "failed to open output file: " +
                output_path.string()
            );
        }

        output << std::setprecision(9);

        constexpr int threads = 256;
        constexpr int warmup = 10;
        constexpr int repeats = 100;
        constexpr int launches_per_sample = 200;
        constexpr float eps = 1e-6f;

        const int row_values[] = {
            1, 32, 256
        };

        const int col_values[] = {
            32, 128, 512, 1024, 2048
        };


        const RmsNormKind kinds[] = {
            RmsNormKind::Baseline,
            RmsNormKind::Block,
        };

        std::vector<BenchmarkResult> results;

        for (int rows : row_values) {
            for (int cols : col_values) {
                for (RmsNormKind kind : kinds) {
                    results.push_back(
                        run_benchmark(
                            rows,
                            cols,
                            eps,
                            threads,
                            kind,
                            warmup,
                            repeats,
                            launches_per_sample
                        )
                    );
                }
            }
        }

        output
            << "{\n"
            << "  \"schema_version\": 2,\n"
            << "  \"benchmark\": \"rmsnorm_f32\",\n"

            << "  \"run_id\": "
            << std::quoted(args.run_id)
            << ",\n"

            << "  \"git_revision\": "
            << std::quoted(args.git_revision)
            << ",\n"

            << "  \"environment\": {\n"

            << "    \"device_ordinal\": "
            << environment.device_ordinal
            << ",\n"

            << "    \"gpu_name\": "
            << std::quoted(environment.gpu_name)
            << ",\n"

            << "    \"compute_capability\": "
            << std::quoted(
                std::to_string(
                    environment.compute_capability_major
                ) +
                "." +
                std::to_string(
                    environment.compute_capability_minor
                )
            )
            << ",\n"

            << "    \"multiprocessor_count\": "
            << environment.multiprocessor_count
            << ",\n"

            << "    \"max_threads_per_block\": "
            << environment.max_threads_per_block
            << ",\n"

            << "    \"total_global_memory_bytes\": "
            << environment.total_global_memory_bytes
            << ",\n"

            << "    \"nvidia_driver_version\": "
            << std::quoted(
                args.nvidia_driver_version
            )
            << ",\n"

            << "    \"cuda_compiled_version\": "
            << std::quoted(
                environment.cuda_compiled_version
            )
            << ",\n"

            << "    \"cuda_runtime_version\": "
            << std::quoted(
                environment.cuda_runtime_version
            )
            << ",\n"

            << "    \"cuda_driver_api_version\": "
            << std::quoted(
                environment.cuda_driver_api_version
            )
            << "\n"

            << "  },\n"

            << "  \"storage_dtype\": \"float32\",\n"
            << "  \"accumulator_dtype\": \"float32\",\n"

            << "  \"eps\": "
            << eps
            << ",\n"

            << "  \"input_pattern\": "
            << std::quoted("sin(index * 0.01)")
            << ",\n"

            << "  \"correctness_check\": "
            << std::quoted(
                "cpu_reference_once_after_timing_per_case"
            )
            << ",\n"

            << "  \"workload_order\": "
            << std::quoted(
                "rows_then_cols_then_baseline_block"
            )
            << ",\n"

            << "  \"measurement\": "
            << std::quoted(
                "batched_average_per_launch"
            )
            << ",\n"

            << "  \"timer\": "
            << std::quoted(
                "cuda_event_same_stream"
            )
            << ",\n"

            << "  \"synchronization\": "
            << std::quoted(
                "cudaEventSynchronize_stop"
            )
            << ",\n"

            << "  \"time_unit\": \"ms\",\n"

            << "  \"warmup_batches\": "
            << warmup
            << ",\n"

            << "  \"repeats\": "
            << repeats
            << ",\n"

            << "  \"launches_per_sample\": "
            << launches_per_sample
            << ",\n"

            << "  \"results\": [\n";

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
                    rmsnorm_kind_name(result.kind)
                )
                << ",\n"
                << "      \"median_per_launch_ms\": "
                << result.median_ms << ",\n"
                << "      \"p95_sample_mean_per_launch_ms\": "
                << result.p95_ms << ",\n"
                << "      \"samples_per_launch_ms\": [";

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

        output.flush();

        if (!output.good()) {
            throw std::runtime_error(
                "failed while writing output file: " +
                output_path.string()
            );
        }

        output.close();

        std::cout
            << "WROTE benchmark JSON: "
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