find_package(Threads REQUIRED)

add_library(inferflow_core
    src/service_core.cpp
    src/serving_engine.cpp
)

target_include_directories(inferflow_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(inferflow_core PUBLIC cxx_std_20)
target_link_libraries(inferflow_core PUBLIC Threads::Threads)

if(BUILD_TESTING)
    add_executable(test_service_core
        tests/test_service_core.cpp
    )

    target_include_directories(test_service_core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    target_link_libraries(test_service_core
        PRIVATE
            inferflow_core
    )

    add_test(
        NAME inferflow_service_core
        COMMAND test_service_core
    )
    add_executable(test_serving_engine tests/test_serving_engine.cpp)
    target_include_directories(test_serving_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_serving_engine PRIVATE inferflow_core)
    add_test(NAME inferflow_serving_engine COMMAND test_serving_engine)

    add_executable(test_dynamic_batch tests/test_dynamic_batch.cpp)
    target_include_directories(test_dynamic_batch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(test_dynamic_batch PRIVATE inferflow_core)
    add_test(NAME inferflow_dynamic_batch COMMAND test_dynamic_batch)
endif()
