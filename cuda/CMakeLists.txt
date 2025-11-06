cmake_minimum_required(VERSION 4.1)
project(gen_meta_balls VERSION 0.0.1 LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# TODO: add the rest of the requirements


add_subdirectory(deps)
