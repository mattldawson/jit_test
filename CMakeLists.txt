cmake_minimum_required(VERSION 3.4.3)
project(JIT_TEST)

################################################################################
# llvm

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})



################################################################################
# jit-test
set (CMAKE_CXX_STANDARD 11)
file(GLOB JIT_TEST_SRCS src/*.cpp)

add_executable(jit_test ${JIT_TEST_SRCS})

llvm_map_components_to_libnames(llvm_libs support core orcjit native irreader)

target_link_libraries(jit_test ${llvm_libs})



################################################################################
