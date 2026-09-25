cmake_minimum_required(VERSION 3.26)
project(edgeai_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _C NB_STATIC
  src/bindings.cpp
  src/isa.cpp
  src/gemm_f32.cpp
  src/gemm_s8.cpp
  src/gemm_w4.cpp
  src/gemm_sparse.cpp
  src/im2col_quant.cpp
  src/microbench.cpp)
target_include_directories(_C PRIVATE include)
target_compile_options(_C PRIVATE -Wall -Wextra -Wno-unused-parameter)

# x86 SIMD variants use per-function target attributes and runtime dispatch, so
# the baseline stays generic x86-64. On aarch64, dotprod is the baseline (all
# Apple Silicon; Linux builds check HWCAP at runtime before using it).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND NOT APPLE)
  target_compile_options(_C PRIVATE -march=armv8.2-a+dotprod)
endif()

# Install next to the Python package: edge_ai_compression/inference/_C*.so
install(TARGETS _C LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../edge_ai_compression/inference")
