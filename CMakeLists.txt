cmake_minimum_required(VERSION 3.20)
project(jax_mps LANGUAGES C CXX OBJCXX)

# Add default path for MLIR/StableHLO dependencies
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/jax-mps-deps")

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

# XLA PJRT headers (cloned into third_party/xla)
set(XLA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/xla)

# Find Protobuf and Abseil (static builds from setup_deps.sh for wheel distribution)
# Prefer static libraries to avoid runtime dependencies
set(CMAKE_FIND_LIBRARY_SUFFIXES_ORIG ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package(absl REQUIRED)

# Find protoc executable from our prefix path before finding Protobuf package
find_program(Protobuf_PROTOC_EXECUTABLE protoc
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
)
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_VERSION}")
message(STATUS "Found protoc: ${Protobuf_PROTOC_EXECUTABLE}")

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_ORIG})

# Generate C++ from proto files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/proto/device_assignment.proto)

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

# Find StableHLO headers
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
find_library(STABLEHLO_BROADCAST_UTILS_LIB StablehloBroadcastUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_BROADCAST_LOWERING_LIB StablehloBroadcastLowering HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_PASSES_LIB StablehloPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_OPTIMIZATION_PASSES_LIB StablehloOptimizationPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_PASS_UTILS_LIB StablehloPassUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_TYPE_CONVERSION_LIB StablehloTypeConversion HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(VERSION_LIB Version HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)

# Source files
set(PJRT_SOURCES
    src/pjrt_plugin/pjrt_api.cc
    src/pjrt_plugin/pjrt_client.cc
    src/pjrt_plugin/pjrt_device.cc
    src/pjrt_plugin/pjrt_memory.cc
    src/pjrt_plugin/pjrt_buffer.cc
    src/pjrt_plugin/pjrt_executable.cc
    src/pjrt_plugin/pjrt_event.cc
    src/pjrt_plugin/pjrt_topology.cc
    src/pjrt_plugin/mps_client.mm
    src/pjrt_plugin/mps_device.mm
    src/pjrt_plugin/mps_buffer.mm
    src/pjrt_plugin/mps_executable.mm
    src/pjrt_plugin/stablehlo_parser.mm
    src/pjrt_plugin/type_utils.mm
    src/pjrt_plugin/ops/binary_ops.mm
    src/pjrt_plugin/ops/unary_ops.mm
    src/pjrt_plugin/ops/shape_ops.mm
    src/pjrt_plugin/ops/bitwise_ops.mm
    src/pjrt_plugin/ops/convolution_ops.mm
    src/pjrt_plugin/ops/reduction_ops.mm
    ${PROTO_SRCS}
)

add_library(pjrt_plugin_mps SHARED ${PJRT_SOURCES})

target_include_directories(pjrt_plugin_mps PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${XLA_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${STABLEHLO_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated proto headers
    ${Protobuf_INCLUDE_DIRS}
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

# Abseil libraries needed by protobuf and our code (for static linking)
set(ABSL_LIBS
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    # Required by protobuf when statically linked
    absl::cord
    absl::strings
    absl::str_format
    absl::synchronization
    absl::hash
    absl::raw_hash_set
    absl::status
    absl::statusor
    absl::span
    absl::optional
    absl::variant
    absl::any
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
)

# Find utf8_range libraries (required by protobuf)
find_library(UTF8_RANGE_LIB utf8_range HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(UTF8_VALIDITY_LIB utf8_validity HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
set(UTF8_LIBS "")
if(UTF8_RANGE_LIB)
    list(APPEND UTF8_LIBS ${UTF8_RANGE_LIB})
endif()
if(UTF8_VALIDITY_LIB)
    list(APPEND UTF8_LIBS ${UTF8_VALIDITY_LIB})
endif()

target_link_libraries(pjrt_plugin_mps PRIVATE
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${STABLEHLO_LIBS}
    ${MLIR_LIBS}
    protobuf::libprotobuf
    ${UTF8_LIBS}
    ${ABSL_LIBS}
)

# For wheel distribution: ensure RPATH is set correctly
set_target_properties(pjrt_plugin_mps PROPERTIES
    OUTPUT_NAME "pjrt_plugin_mps"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install target - this is used by scikit-build-core
# Note: scikit-build's wheel.install-dir adds "jax_plugins/mps" prefix,
# so we only need "lib" here to get "jax_plugins/mps/lib" in the wheel
install(TARGETS pjrt_plugin_mps
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)
