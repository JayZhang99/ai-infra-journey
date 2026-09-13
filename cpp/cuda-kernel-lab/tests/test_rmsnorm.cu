#include "rmsnorm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <limits>

namespace {

template <typename Fn>
void expect_invalid_argument(
    const std::string& case_name,
    Fn&& fn
) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::invalid_argument&) {
        std::cout
            << "PASSED invalid case: "
            << case_name
            << '\n';
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            case_name +
            " threw wrong exception type: " +
            error.what()
        );
    }

    throw std::runtime_error(
        case_name +
        " did not throw std::invalid_argument"
    );
}

struct TestDeviceResources {
    float* input = nullptr;
    float* weight = nullptr;
    float* output = nullptr;
    cudaStream_t stream = nullptr;

    ~TestDeviceResources() {
        if (output != nullptr) {
            cudaFree(output);
        }
        if (weight != nullptr) {
            cudaFree(weight);
        }
        if (input != nullptr) {
            cudaFree(input);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
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

void test_zero_input(
    int rows,
    int cols,
    RmsNormKind kind
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;

    const std::size_t tensor_bytes =
        count * sizeof(float);

    const std::size_t weight_bytes =
        static_cast<std::size_t>(cols) *
        sizeof(float);

    std::vector<float> input(count, 0.0f);
    std::vector<float> weight(cols);
    std::vector<float> actual(
        count,
        std::numeric_limits<float>::quiet_NaN()
    );

    for (int c = 0; c < cols; ++c) {
        weight[c] =
            0.5f +
            0.01f * static_cast<float>(c % 17);
    }

    TestDeviceResources device;

    cuda_check(
        cudaStreamCreate(&device.stream),
        "create zero-case stream"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.input),
            tensor_bytes
        ),
        "allocate zero-case input"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.weight),
            weight_bytes
        ),
        "allocate zero-case weight"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.output),
            tensor_bytes
        ),
        "allocate zero-case output"
    );

    cuda_check(
        cudaMemcpyAsync(
            device.input,
            input.data(),
            tensor_bytes,
            cudaMemcpyHostToDevice,
            device.stream
        ),
        "copy zero input"
    );

    cuda_check(
        cudaMemcpyAsync(
            device.weight,
            weight.data(),
            weight_bytes,
            cudaMemcpyHostToDevice,
            device.stream
        ),
        "copy zero-case weight"
    );

    // 先写入 NaN，用来检测 Kernel 是否漏写输出。
    cuda_check(
        cudaMemcpyAsync(
            device.output,
            actual.data(),
            tensor_bytes,
            cudaMemcpyHostToDevice,
            device.stream
        ),
        "initialize zero-case output"
    );

    launch_rmsnorm_f32(
        device.input,
        device.weight,
        device.output,
        rows,
        cols,
        1e-6f,
        256,
        kind,
        device.stream
    );

    cuda_check(
        cudaMemcpyAsync(
            actual.data(),
            device.output,
            tensor_bytes,
            cudaMemcpyDeviceToHost,
            device.stream
        ),
        "copy zero-case output"
    );

    cuda_check(
        cudaStreamSynchronize(device.stream),
        "synchronize zero case"
    );

    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error(
                "zero input produced non-finite value "
                "at index " +
                std::to_string(i)
            );
        }

        if (actual[i] != 0.0f) {
            throw std::runtime_error(
                "zero input produced non-zero value "
                "at index " +
                std::to_string(i)
            );
        }
    }

    std::cout
        << "PASSED zero input"
        << " kind=" << rmsnorm_kind_name(kind)
        << " rows=" << rows
        << " cols=" << cols
        << '\n';
}

void test_invalid_arguments() {
    constexpr int buffer_elements = 128;
    constexpr std::size_t bytes =
        buffer_elements * sizeof(float);

    TestDeviceResources device;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.input),
            bytes
        ),
        "allocate contract input"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.weight),
            bytes
        ),
        "allocate contract weight"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device.output),
            bytes
        ),
        "allocate contract output"
    );

    const auto launch = [
        &device
    ](
        int rows,
        int cols,
        float eps,
        int threads,
        RmsNormKind kind
    ) {
        launch_rmsnorm_f32(
            device.input,
            device.weight,
            device.output,
            rows,
            cols,
            eps,
            threads,
            kind
        );
    };

    // rows 合同
    expect_invalid_argument(
        "rows equals zero",
        [&] {
            launch(
                0, 32, 1e-6f, 256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "rows is negative",
        [&] {
            launch(
                -1, 32, 1e-6f, 256,
                RmsNormKind::Block
            );
        }
    );

    // cols 合同
    expect_invalid_argument(
        "cols equals zero",
        [&] {
            launch(
                1, 0, 1e-6f, 256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "cols is negative",
        [&] {
            launch(
                1, -1, 1e-6f, 256,
                RmsNormKind::Block
            );
        }
    );

    // eps 合同
    expect_invalid_argument(
        "eps equals zero",
        [&] {
            launch(
                1, 32, 0.0f, 256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "eps is negative",
        [&] {
            launch(
                1, 32, -1e-6f, 256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "eps is NaN",
        [&] {
            launch(
                1,
                32,
                std::numeric_limits<float>::
                    quiet_NaN(),
                256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "eps is positive infinity",
        [&] {
            launch(
                1,
                32,
                std::numeric_limits<float>::
                    infinity(),
                256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "eps is negative infinity",
        [&] {
            launch(
                1,
                32,
                -std::numeric_limits<float>::
                    infinity(),
                256,
                RmsNormKind::Block
            );
        }
    );

    // threads 合同
    expect_invalid_argument(
        "threads equals zero",
        [&] {
            launch(
                1, 32, 1e-6f, 0,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "threads below one warp",
        [&] {
            launch(
                1, 32, 1e-6f, 31,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "threads is not multiple of 32",
        [&] {
            launch(
                1, 32, 1e-6f, 33,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "threads exceeds 1024",
        [&] {
            launch(
                1, 32, 1e-6f, 1056,
                RmsNormKind::Block
            );
        }
    );

    // 枚举合同
    expect_invalid_argument(
        "unsupported RmsNormKind",
        [&] {
            launch(
                1,
                32,
                1e-6f,
                256,
                static_cast<RmsNormKind>(999)
            );
        }
    );

    // 每个指针应单独测试，不能只传三个 nullptr。
    expect_invalid_argument(
        "null input",
        [&] {
            launch_rmsnorm_f32(
                nullptr,
                device.weight,
                device.output,
                1,
                32,
                1e-6f,
                256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "null weight",
        [&] {
            launch_rmsnorm_f32(
                device.input,
                nullptr,
                device.output,
                1,
                32,
                1e-6f,
                256,
                RmsNormKind::Block
            );
        }
    );

    expect_invalid_argument(
        "null output",
        [&] {
            launch_rmsnorm_f32(
                device.input,
                device.weight,
                nullptr,
                1,
                32,
                1e-6f,
                256,
                RmsNormKind::Block
            );
        }
    );
}

std::vector<float> cpu_rmsnorm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int rows,
    int cols,
    float eps
) {
    std::vector<float> output(input.size());
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

    return output;
}

void run_case(
    int rows,
    int cols,
    RmsNormKind kind
) {
    const std::size_t count =
        static_cast<std::size_t>(rows) * cols;
    const std::size_t tensor_bytes =
        static_cast<std::size_t>(rows) * cols * sizeof(float);

    const std::size_t weight_bytes =
        static_cast<std::size_t>(cols) * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> weight(cols);

    float eps = 1e-6;

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
        weight[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    }



    const std::vector<float> expected = cpu_rmsnorm(input, weight, rows, cols, eps);

    float* device_input = nullptr;
    float* device_output = nullptr;
    float* device_weight = nullptr;
    cudaStream_t stream = nullptr;

    cuda_check(
        cudaStreamCreate(&stream),
        "create stream"
    );
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
        "synchronize rmsnorm"
    );

    std::vector<float> actual(count);

    cuda_check(
        cudaMemcpy(
            actual.data(),
            device_output,
            tensor_bytes,
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
    cudaStreamDestroy(stream);
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

        const RmsNormKind kinds[] = {
            RmsNormKind::Baseline,
            RmsNormKind::Block,
        };

        for (RmsNormKind kind : kinds) {
            for (const auto& [rows, cols] : cases) {
                run_case(rows, cols, kind);
            }
        }

        for (RmsNormKind kind : kinds) {
            test_zero_input(1, 1, kind);
            test_zero_input(1, 33, kind);
            test_zero_input(4, 128, kind);
        }

        test_invalid_arguments();

        std::cout << "PASSED: all rmsnorm checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}