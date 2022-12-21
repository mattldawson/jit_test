cmake_minimum_required(VERSION 3.4.3)
project(JIT_TEST LANGUAGES CXX C Fortran)
set (CMAKE_CXX_STANDARD 17)

include(FortranCInterface)

# if you are working with C++ and Fortran
FortranCInterface_VERIFY(CXX)

################################################################################
# LLVM flags

set(LLVM_CMD "llvm-config --cxxflags --ldflags --system-libs --libs support core orcjit native irreader | tr '\\n' ' '")
execute_process(COMMAND bash "-c" ${LLVM_CMD}
                OUTPUT_VARIABLE llvm_libs)
separate_arguments(llvm_libs)

################################################################################
# Create preprocessed fortran



################################################################################
# JIT tests

enable_testing( )

add_subdirectory(test)

################################################################################
