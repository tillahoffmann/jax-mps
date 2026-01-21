cmake_minimum_required(VERSION 3.20)
project(jax_mps LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# macOS specific settings
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

# Option to use MLIR for proper StableHLO parsing
option(JAX_MPS_USE_MLIR "Use MLIR/StableHLO libraries for proper parsing" ON)

# Find required Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Use vendored XLA PJRT headers
set(XLA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party)

# Source files
set(PJRT_SOURCES
    src/pjrt_plugin/pjrt_api.cc
    src/pjrt_plugin/mps_client.mm
    src/pjrt_plugin/mps_device.mm
    src/pjrt_plugin/mps_buffer.mm
    src/pjrt_plugin/mps_executable.mm
    src/mps_backend/mps_ops.mm
)

if(JAX_MPS_USE_MLIR)
    # Find MLIR and StableHLO packages
    # Set CMAKE_PREFIX_PATH to the install prefix from setup_deps.sh
    find_package(MLIR REQUIRED CONFIG)
    find_package(StableHLO REQUIRED CONFIG)

    message(STATUS "Found MLIR: ${MLIR_DIR}")
    message(STATUS "Found StableHLO: ${StableHLO_DIR}")

    # Include MLIR/LLVM cmake modules
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    include(AddLLVM)
    include(AddMLIR)

    # Add MLIR-based parser
    list(APPEND PJRT_SOURCES src/pjrt_plugin/mlir_parser.cc)

    add_library(pjrt_plugin_mps SHARED ${PJRT_SOURCES})

    target_compile_definitions(pjrt_plugin_mps PRIVATE
        JAX_MPS_USE_MLIR=1
    )

    target_include_directories(pjrt_plugin_mps PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${XLA_INCLUDE_DIR}
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )

    # Get required MLIR libraries
    get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

    target_link_libraries(pjrt_plugin_mps PRIVATE
        ${METAL_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${MPSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        # MLIR core
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRBytecodeReader
        MLIRBytecodeWriter
        # StableHLO
        StablehloOps
        StablehloSerialization
        VhloOps
        ChloOps
    )

else()
    # Lightweight build: use hand-rolled parser + subprocess fallback
    message(STATUS "Building without MLIR (lightweight mode)")
    message(STATUS "For proper StableHLO parsing, run ./scripts/setup_deps.sh")

    list(APPEND PJRT_SOURCES src/pjrt_plugin/stablehlo_parser.cc)

    add_library(pjrt_plugin_mps SHARED ${PJRT_SOURCES})

    target_compile_definitions(pjrt_plugin_mps PRIVATE
        JAX_MPS_USE_MLIR=0
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
endif()

# Set output name
set_target_properties(pjrt_plugin_mps PROPERTIES
    OUTPUT_NAME "pjrt_plugin_mps"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install target
install(TARGETS pjrt_plugin_mps
    LIBRARY DESTINATION lib
)
