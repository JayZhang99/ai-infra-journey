#include "transpose.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>
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

struct DeviceBuffer {
    float* ptr = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&ptr),
                bytes
            ),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(
        const DeviceBuffer&
    ) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;

    Stream() {
        cuda_check(
            cudaStreamCreate(&value),
            "cudaStreamCreate"
        );
    }

    ~Stream() {
        if (value != nullptr) {
            cudaStreamDestroy(value);
        }
    }
};

std::vector<float> cpu_transpose(
    const std::vector<float>& input,
    int height,
    int width
) {
    std::vector<float> output(
        static_cast<std::size_t>(height) *
        width
    );

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            output[col * height + row] =
                input[row * width + col];
        }
    }

    return output;
}

void run_case(
    int height,
    int width,
    TransposeKind kind
) {
    const std::size_t count =
        static_cast<std::size_t>(height) *
        width;

    const std::size_t bytes =
        count * sizeof(float);

    std::vector<float> input(count);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const int raw =
                (row * 37 + col * 13) % 257 -
                128;

            input[row * width + col] =
                static_cast<float>(raw) *
                0.125f;
        }
    }

    const std::vector<float> expected =
        cpu_transpose(input, height, width);

    // 先填 NaN，用来发现漏写。
    std::vector<float> actual(
        count,
        std::numeric_limits<float>::
            quiet_NaN()
    );

    DeviceBuffer device_input(bytes);
    DeviceBuffer device_output(bytes);
    Stream stream;

    cuda_check(
        cudaMemcpyAsync(
            device_input.ptr,
            input.data(),
            bytes,
            cudaMemcpyHostToDevice,
            stream.value
        ),
        "copy input"
    );

    cuda_check(
        cudaMemcpyAsync(
            device_output.ptr,
            actual.data(),
            bytes,
            cudaMemcpyHostToDevice,
            stream.value
        ),
        "initialize output"
    );

    launch_transpose_f32(
        device_input.ptr,
        device_output.ptr,
        height,
        width,
        256,
        kind,
        stream.value
    );

    cuda_check(
        cudaMemcpyAsync(
            actual.data(),
            device_output.ptr,
            bytes,
            cudaMemcpyDeviceToHost,
            stream.value
        ),
        "copy output"
    );

    cuda_check(
        cudaStreamSynchronize(stream.value),
        "synchronize"
    );

    // Transpose 没有算术运算，应当精确相等。
    for (std::size_t i = 0; i < count; ++i) {
        if (
            !std::isfinite(actual[i]) ||
            actual[i] != expected[i]
        ) {
            throw std::runtime_error(
                "transpose mismatch"
                " kind=" +
                std::string(
                    transpose_kind_name(kind)
                ) +
                " height=" +
                std::to_string(height) +
                " width=" +
                std::to_string(width) +
                " index=" +
                std::to_string(i) +
                " expected=" +
                std::to_string(expected[i]) +
                " actual=" +
                std::to_string(actual[i])
            );
        }
    }

    std::cout
        << "PASSED"
        << " kind=" << transpose_kind_name(kind)
        << " height=" << height
        << " width=" << width
        << '\n';
}

template <typename Fn>
void expect_invalid(
    const std::string& name,
    Fn&& fn
) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::invalid_argument&) {
        std::cout
            << "PASSED invalid case: "
            << name
            << '\n';
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            name +
            " threw wrong exception: " +
            error.what()
        );
    }

    throw std::runtime_error(
        name + " was not rejected"
    );
}

void test_invalid_arguments() {
    constexpr std::size_t bytes =
        1024 * sizeof(float);

    DeviceBuffer input(bytes);
    DeviceBuffer output(bytes);

    const auto launch = [
        &input,
        &output
    ](
        int height,
        int width,
        int threads,
        TransposeKind kind
    ) {
        launch_transpose_f32(
            input.ptr,
            output.ptr,
            height,
            width,
            threads,
            kind
        );
    };

    expect_invalid("height zero", [&] {
        launch(
            0, 32, 256,
            TransposeKind::Naive
        );
    });

    expect_invalid("height negative", [&] {
        launch(
            -1, 32, 256,
            TransposeKind::Naive
        );
    });

    expect_invalid("width zero", [&] {
        launch(
            32, 0, 256,
            TransposeKind::Naive
        );
    });

    expect_invalid("threads 128", [&] {
        launch(
            32, 32, 128,
            TransposeKind::Tiled
        );
    });

    expect_invalid("threads 512", [&] {
        launch(
            32, 32, 512,
            TransposeKind::Padding
        );
    });

    expect_invalid("unsupported kind", [&] {
        launch(
            32,
            32,
            256,
            static_cast<TransposeKind>(999)
        );
    });

    expect_invalid("null input", [&] {
        launch_transpose_f32(
            nullptr,
            output.ptr,
            32,
            32,
            256,
            TransposeKind::Naive
        );
    });

    expect_invalid("null output", [&] {
        launch_transpose_f32(
            input.ptr,
            nullptr,
            32,
            32,
            256,
            TransposeKind::Naive
        );
    });
}

}  // namespace

int main() {
    try {
        int device_count = 0;

        const cudaError_t status =
            cudaGetDeviceCount(&device_count);

        if (
            status != cudaSuccess ||
            device_count == 0
        ) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }

        const std::vector<
            std::pair<int, int>
        > shapes = {
            {1, 1},
            {1, 33},
            {31, 33},
            {32, 32},
            {33, 65},
            {127, 129},
        };

        const TransposeKind kinds[] = {
            TransposeKind::Naive,
            TransposeKind::Tiled,
            TransposeKind::Padding,
        };

        for (TransposeKind kind : kinds) {
            for (
                const auto& [height, width] :
                shapes
            ) {
                run_case(
                    height,
                    width,
                    kind
                );
            }
        }

        test_invalid_arguments();

        std::cout
            << "PASSED: all transpose checks\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "FAILED: "
            << error.what()
            << '\n';

        return 1;
    }
}