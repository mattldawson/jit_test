################################################################################
# Test of chemical forcing (matrix_multiply) function

file(GLOB MAT_MUL_TEST_SRCS *.c*)

add_executable(matrix_multiply_test ${MAT_MUL_TEST_SRCS})

target_include_directories(matrix_multiply_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(matrix_multiply_test ${llvm_libs})

add_test(NAME matrix_multiply COMMAND matrix_multiply_test)

################################################################################
