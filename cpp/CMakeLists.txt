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
