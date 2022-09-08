cmake_minimum_required(VERSION 3.4.3)
project(JIT_TEST)

################################################################################
# jit-test

set (CMAKE_CXX_STANDARD 14)
file(GLOB JIT_TEST_SRCS src/*.cpp)

add_executable(jit_test ${JIT_TEST_SRCS})

set(LLVM_CMD "llvm-config --cxxflags --ldflags --system-libs --libs support core orcjit native irreader | tr '\\n' ' '")
execute_process(COMMAND bash "-c" ${LLVM_CMD}
                OUTPUT_VARIABLE llvm_libs)
separate_arguments(llvm_libs)

target_link_libraries(jit_test ${llvm_libs})



################################################################################
