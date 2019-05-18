cmake_minimum_required(VERSION 3.1)
project(JIT_TEST)

################################################################################
# jit-test
set (CMAKE_CXX_STANDARD 11)
file(GLOB JIT_TEST_SRCS src/*.cpp)

add_executable(jit_test ${JIT_TEST_SRCS})

################################################################################
