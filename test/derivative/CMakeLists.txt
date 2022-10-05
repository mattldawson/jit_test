################################################################################
# Test of chemical forcing (derivative) function

file(GLOB DERIV_TEST_SRCS *.cpp)

add_executable(derivative_test ${DERIV_TEST_SRCS})

target_include_directories(derivative_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(derivative_test ${llvm_libs})

add_test(NAME derivative COMMAND derivative_test)

################################################################################
