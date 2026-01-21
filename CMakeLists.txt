cmake_minimum_required(VERSION 3.20)
project(jax_mps LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# macOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

# Find required Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Use vendored XLA PJRT headers
set(XLA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party)

# PJRT Plugin shared library
# Note: We use stablehlo-translate subprocess for bytecode parsing
# to avoid linking against the large MLIR/StableHLO libraries
add_library(pjrt_plugin_mps SHARED
    src/pjrt_plugin/pjrt_api.cc
    src/pjrt_plugin/mps_client.mm
    src/pjrt_plugin/mps_device.mm
    src/pjrt_plugin/mps_buffer.mm
    src/pjrt_plugin/mps_executable.mm
    src/pjrt_plugin/stablehlo_parser.cc
    src/mps_backend/mps_ops.mm
)

target_include_directories(pjrt_plugin_mps PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${XLA_INCLUDE_DIR}
)

target_link_libraries(pjrt_plugin_mps PRIVATE
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Set output name
set_target_properties(pjrt_plugin_mps PROPERTIES
    OUTPUT_NAME "pjrt_plugin_mps"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install target
install(TARGETS pjrt_plugin_mps
    LIBRARY DESTINATION lib
)
