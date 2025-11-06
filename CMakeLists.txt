cmake_minimum_required(VERSION 4.1)
project(genmetaballs VERSION 0.0.1 LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)



add_library(genmetaballs_core)

# TODO: add the headers and sources under `genmetaballs/src/cuda/core` to the
# `gen_meta_balls_core` library.

# TODO: make any fixes and improvements if necessary for CUDA integration

find_package(Python 3.13 
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# Build the extension
nanobind_add_module(
  _genmetaballs_bindings
  NB_STATIC
  STABLE_ABI
  src/cuda/bindings.cu
)

# Install the extension into the Python package directory
install(TARGETS _genmetaballs_bindings LIBRARY DESTINATION genmetaballs)
