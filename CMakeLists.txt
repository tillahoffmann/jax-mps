cmake_minimum_required(VERSION 3.20)
project(jax_mps LANGUAGES C CXX OBJCXX)

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

# Source files (common to both builds)
set(PJRT_SOURCES
    src/pjrt_plugin/pjrt_api.cc
    src/pjrt_plugin/mps_client.mm
    src/pjrt_plugin/mps_device.mm
    src/pjrt_plugin/mps_buffer.mm
    src/pjrt_plugin/mps_executable.mm
    src/mps_backend/mps_ops.mm
)

if(JAX_MPS_USE_MLIR)
    message(STATUS "Building with MLIR/StableHLO support")

    # Find MLIR and LLVM packages
    find_package(MLIR REQUIRED CONFIG)
    find_package(LLVM REQUIRED CONFIG)

    message(STATUS "Found MLIR: ${MLIR_DIR}")
    message(STATUS "Found LLVM: ${LLVM_DIR}")

    # Include MLIR/LLVM cmake modules
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

    include(AddLLVM)
    include(AddMLIR)

    # StableHLO doesn't provide CMake configs, find manually
    # Use CMAKE_PREFIX_PATH to locate stablehlo headers and libs
    find_path(STABLEHLO_INCLUDE_DIR
        NAMES stablehlo/dialect/StablehloOps.h
        HINTS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES include
    )
    if(STABLEHLO_INCLUDE_DIR)
        message(STATUS "Found StableHLO headers: ${STABLEHLO_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "StableHLO headers not found. Run scripts/setup_deps.sh first.")
    endif()

    # Find StableHLO static libraries
    find_library(STABLEHLO_OPS_LIB StablehloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_SERIALIZATION_LIB StablehloSerialization HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(VHLO_OPS_LIB VhloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(CHLO_OPS_LIB ChloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_BASE_LIB StablehloBase HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_REGISTER_LIB StablehloRegister HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(VHLO_TYPES_LIB VhloTypes HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_TYPE_INFERENCE_LIB StablehloTypeInference HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_ASSEMBLY_FORMAT_LIB StablehloAssemblyFormat HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)

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
        ${STABLEHLO_INCLUDE_DIR}
    )

    # Core MLIR libraries (order matters for static linking)
    set(MLIR_LIBS
        MLIRReconcileUnrealizedCasts
        MLIRPass
        MLIRTransforms
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRArithDialect
        MLIRComplexDialect
        MLIRQuantDialect
        MLIRShapeDialect
        MLIRTensorDialect
        MLIRDataLayoutInterfaces
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
        MLIRTransformUtils
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRSupport
        LLVMSupport
    )

    # Find additional StableHLO libraries
    find_library(STABLEHLO_BROADCAST_UTILS_LIB StablehloBroadcastUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_BROADCAST_LOWERING_LIB StablehloBroadcastLowering HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_PASSES_LIB StablehloPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_OPTIMIZATION_PASSES_LIB StablehloOptimizationPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_PASS_UTILS_LIB StablehloPassUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(STABLEHLO_TYPE_CONVERSION_LIB StablehloTypeConversion HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    find_library(VERSION_LIB Version HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)

    # Collect found StableHLO libraries
    set(STABLEHLO_LIBS "")
    foreach(lib
        STABLEHLO_SERIALIZATION_LIB
        STABLEHLO_PASSES_LIB
        STABLEHLO_OPTIMIZATION_PASSES_LIB
        STABLEHLO_PASS_UTILS_LIB
        STABLEHLO_TYPE_CONVERSION_LIB
        STABLEHLO_OPS_LIB
        VHLO_OPS_LIB
        CHLO_OPS_LIB
        STABLEHLO_BASE_LIB
        STABLEHLO_REGISTER_LIB
        VHLO_TYPES_LIB
        STABLEHLO_TYPE_INFERENCE_LIB
        STABLEHLO_ASSEMBLY_FORMAT_LIB
        STABLEHLO_BROADCAST_UTILS_LIB
        STABLEHLO_BROADCAST_LOWERING_LIB
        VERSION_LIB)
        if(${lib})
            list(APPEND STABLEHLO_LIBS ${${lib}})
        endif()
    endforeach()

    message(STATUS "StableHLO libraries: ${STABLEHLO_LIBS}")

    target_link_libraries(pjrt_plugin_mps PRIVATE
        ${METAL_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${MPSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${STABLEHLO_LIBS}
        ${MLIR_LIBS}
    )

    # For wheel distribution: ensure RPATH is set correctly
    set_target_properties(pjrt_plugin_mps PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

else()
    # Lightweight build: use hand-rolled parser + subprocess fallback
    message(STATUS "Building without MLIR (lightweight mode)")

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

# Common target properties
set_target_properties(pjrt_plugin_mps PROPERTIES
    OUTPUT_NAME "pjrt_plugin_mps"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install target - this is used by scikit-build-core
install(TARGETS pjrt_plugin_mps
    LIBRARY DESTINATION jax_plugins/mps/lib
    RUNTIME DESTINATION jax_plugins/mps/lib
)
