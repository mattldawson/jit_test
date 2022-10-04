################################################################################
# Test of chemical forcing (derivative) function

file(GLOB DERIV_TEST_SRCS *.cpp)

add_executable(derivative_test ${DERIV_TEST_SRCS})

set(LLVM_CMD "llvm-config --cxxflags --ldflags --system-libs --libs support core orcjit native irreader | tr '\\n' ' '")
execute_process(COMMAND bash "-c" ${LLVM_CMD}
                OUTPUT_VARIABLE llvm_libs)
separate_arguments(llvm_libs)

target_include_directories(derivative_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(derivative_test ${llvm_libs})

add_test(NAME derivative COMMAND derivative_test)

################################################################################
