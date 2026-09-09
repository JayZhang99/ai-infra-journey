cmake_minimum_required(VERSION 3.20)
project(ai_infra_journey LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ai_infra_hello src/main.cpp)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME hello_runs COMMAND ai_infra_hello)
endif()

include(CheckLanguage)

option(BUILD_CUDA_LAB "Build CUDA kernel experiments" ON)

if(BUILD_CUDA_LAB)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_subdirectory(cuda-kernel-lab)
    else()
        message(FATAL_ERROR "CUDA compiler not found")
    endif()
endif()
