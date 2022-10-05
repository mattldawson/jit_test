cmake_minimum_required(VERSION 3.4.3)
project(JIT_TEST)
set (CMAKE_CXX_STANDARD 14)

################################################################################
# LLVM flags

set(LLVM_CMD "llvm-config --cxxflags --ldflags --system-libs --libs support core orcjit native irreader | tr '\\n' ' '")
execute_process(COMMAND bash "-c" ${LLVM_CMD}
                OUTPUT_VARIABLE llvm_libs)
separate_arguments(llvm_libs)

################################################################################
# JIT tests

enable_testing( )

add_subdirectory(test)

################################################################################
