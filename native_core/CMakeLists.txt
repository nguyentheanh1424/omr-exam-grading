cmake_minimum_required(VERSION 3.20)

project(omr_native_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(omr_core SHARED
    src/omr_api.cpp
    src/idw_refine.cpp
    src/marker_detect.cpp
    src/region_refine.cpp
    src/warp_global.cpp
)

target_compile_definitions(omr_core PRIVATE OMR_BUILD_DLL=1)
target_include_directories(omr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(omr_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(omr_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(omr_minimal_example
    examples/minimal_usage.c
)
target_include_directories(omr_minimal_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(omr_minimal_example PRIVATE omr_core)

enable_testing()
add_executable(omr_unit_tests
    tests/test_runner.cpp
)
target_include_directories(omr_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(omr_unit_tests PRIVATE omr_core)
add_test(NAME omr_unit_tests COMMAND omr_unit_tests)

install(TARGETS omr_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/omr_api.h DESTINATION include)
