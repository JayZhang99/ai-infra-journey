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

add_library(softmax_kernels STATIC
    src/softmax.cu
)

target_include_directories(softmax_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(softmax_kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(softmax_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(softmax_bench
    src/softmax_bench.cu
)

target_link_libraries(softmax_bench
    PRIVATE
        softmax_kernels
)

if(BUILD_TESTING)
    add_executable(test_softmax
        tests/test_softmax.cu
    )

    target_link_libraries(test_softmax
        PRIVATE
            softmax_kernels
    )

    add_test(
        NAME softmax_correctness
        COMMAND test_softmax
    )

    set_tests_properties(
        softmax_correctness
        PROPERTIES SKIP_RETURN_CODE 77
    )
endif()