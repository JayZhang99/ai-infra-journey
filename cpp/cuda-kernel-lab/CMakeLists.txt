add_library(reduction_kernels STATIC
    src/reduction.cu
)

target_include_directories(reduction_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(reduction_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(reduction_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(reduce_bench
    src/reduce_bench.cu
)

target_link_libraries(reduce_bench
    PRIVATE
        reduction_kernels
)

set_target_properties(reduce_bench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(BUILD_TESTING)
    add_executable(test_reduction
        tests/test_reduction.cu
    )

    target_link_libraries(test_reduction
        PRIVATE
            reduction_kernels
    )

    set_target_properties(test_reduction PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    add_test(
        NAME reduction_correctness
        COMMAND test_reduction
    )
endif()