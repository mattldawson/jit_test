cmake_minimum_required(VERSION 3.21)
project(JIT_TEST LANGUAGES CXX C Fortran)

# connect fortran and c++
# https://enccs.github.io/cmake-workshop/cxx-fortran/
# https://github.com/ENCCS/cmake-workshop/blob/e47ac9b5197fc18db513e62ce3f1ac36357d3839/content/code/day-2/25_cxx-fortran/solution/src/math/CMakeLists.txt
include(FortranCInterface)

include(CMakeDependentOption)

# if you are working with C++ and Fortran
FortranCInterface_VERIFY(CXX)

################################################################################
# Options

option(ENABLE_LLVM "Enable support for LLVM JIT" ON)
if(ENABLE_LLVM)
  set (CMAKE_CXX_STANDARD 17)
  add_compile_definitions(USE_LLVM)
endif()
cmake_dependent_option(ENABLE_GPU "Enable support for GPUs" ON "NOT ENABLE_LLVM" OFF)
cmake_dependent_option(ENABLE_COMPILED "Include compiled GPU kernels in comparison" OFF "ENABLE_GPU" OFF)
if(ENABLE_GPU)
  set (CMAKE_CXX_STANDARD 11)
  enable_language(CUDA)
  add_compile_definitions(USE_GPU)
endif()
if(ENABLE_COMPILED)
  add_compile_definitions(USE_COMPILED)
endif()

set(CUDA_THREADS "128" CACHE STRING "CUDA threads")
set(NCELLS "100" CACHE STRING "Number of model grid cells")
set(NREACTIONS "500" CACHE STRING "Number of reactions to simulate")
set(NSPECIES "200" CACHE STRING "Number of species to simulate")
set(NITERATIONS "1" CACHE STRING "Number of iterations to run the simulate")
math(EXPR CUDA_BLOCKS "(${NCELLS} + ${CUDA_THREADS} - 1) / ${CUDA_THREADS}")
message(STATUS "Running with ${CUDA_BLOCKS} blocks")
add_compile_definitions(NUM_RXNS=${NREACTIONS} NUM_SPEC=${NSPECIES} NUM_REPEAT=${NITERATIONS}
                        CUDA_THREADS=${CUDA_THREADS} CUDA_BLOCKS=${CUDA_BLOCKS} NUM_CELLS=${NCELLS})

################################################################################
# LLVM flags

if(ENABLE_LLVM)
  find_package(LLVM REQUIRED CONFIG
    PATHS
      /opt/homebrew/Cellar/llvm@13/13.0.1_2/
      /glade/work/benkirk/my_spack_playground/deploy/view/CentOS-compilers/llvm/15.0.4/lib/cmake/llvm/
  )
  if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    include_directories(${LLVM_INCLUDE_DIRS})
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})

    llvm_map_components_to_libnames(llvm_libs support core orcjit native irreader)
  else()
    set(LLVM_CMD "llvm-config --cxxflags --ldflags --system-libs --libs support core orcjit native irreader | tr '\\n' ' '")
    execute_process(COMMAND bash "-c" ${LLVM_CMD}
                    OUTPUT_VARIABLE llvm_libs)
    separate_arguments(llvm_libs)
  endif()
endif()

################################################################################
# NVidia libraries

if(ENABLE_GPU)
  SET(CUDA_SEPARABLE_COMPILATION ON)

  find_library(NVRTC_LIB nvrtc
    DOC "NVidia RealTime Compiler library"
    PATHS $ENV{LD_LIBRARY_PATH}
          /glade/u/apps/dav/opt/cuda/11.4.0/lib64/)
  find_library(CUDA_LIB cuda
    DOC "NVidia CUDA library"
    PATHS $ENV{LD_LIBRARY_PATH})
  set(NVIDIA_LIBS ${NVRTC_LIB} ${CUDA_LIB} "-c++libs")

  find_package(OpenACC REQUIRED)
endif( )

################################################################################
# JIT tests

enable_testing( )

add_subdirectory(test)

################################################################################
