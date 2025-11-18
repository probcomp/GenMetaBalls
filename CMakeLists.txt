cmake_minimum_required(VERSION 4.1)
project(genmetaballs VERSION 0.0.1 LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# Generate compile_commands.json for clang-tidy and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################
# Core Library #
################

add_library(genmetaballs_core
  genmetaballs/src/cuda/core/utils.cu
  genmetaballs/src/cuda/core/utils.cuh
  genmetaballs/src/cuda/core/add.cuh
  genmetaballs/src/cuda/core/geometry.cuh
  genmetaballs/src/cuda/core/geometry.cu
)

# Set include directories for the core library
target_include_directories(genmetaballs_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/genmetaballs/src/cuda
)

# Enable Position Independent Code for shared library linking
set_target_properties(genmetaballs_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

############
# Bindings #
############

find_package(Python 3.13
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# Build the extension
nanobind_add_module(
  _genmetaballs_bindings
  NB_STATIC
  STABLE_ABI
  genmetaballs/src/cuda/bindings/main.cu
  genmetaballs/src/cuda/bindings/confidence.cu
)

# Link the core library to the bindings module
target_link_libraries(_genmetaballs_bindings PRIVATE genmetaballs_core)

# Install the extension into the Python package directory
install(TARGETS _genmetaballs_bindings LIBRARY DESTINATION genmetaballs)


################
# Testing      #
################

enable_testing()

# Find GTest
find_package(GTest CONFIG REQUIRED)

# C++/CUDA test executable
file(GLOB CPP_CUDA_TEST_SOURCES tests/cpp_tests/*.cu tests/cpp_tests/*.cpp)
add_executable(test_add ${CPP_CUDA_TEST_SOURCES})

target_link_libraries(test_add
  PRIVATE
    genmetaballs_core
    GTest::gtest
    GTest::gtest_main
)

# Enable CUDA for the test executable
set_target_properties(test_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
gtest_discover_tests(test_add)
