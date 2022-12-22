cmake_minimum_required(VERSION 3.21)
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

# find_package(LLVM REQUIRED CONFIG
#   PATHS
#     /opt/homebrew/Cellar/llvm@13/13.0.1_2/
# )
# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# include_directories(${LLVM_INCLUDE_DIRS})
# separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
# add_definitions(${LLVM_DEFINITIONS_LIST})

# llvm_map_components_to_libnames(llvm_libs support core orcjit native irreader)

################################################################################
# Create preprocessed fortran



################################################################################
# JIT tests

enable_testing( )

add_subdirectory(test)

################################################################################
