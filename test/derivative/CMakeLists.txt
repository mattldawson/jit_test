################################################################################
# Test of chemical forcing (derivative) function

file(GLOB DERIV_TEST_SRCS *.cpp)
list(FILTER DERIV_TEST_SRCS EXCLUDE REGEX "JitTest.cpp")

################################################################################
# Make C++ generate the fortran code
add_executable(generate_fortran GenerateFortran.cpp ClassicDeriv.cpp)
target_include_directories(generate_fortran PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(generate_fortran
  PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
)

# this is what runs the executable to create preprocessed.F90
add_custom_command(
  TARGET generate_fortran
  POST_BUILD
  COMMAND generate_fortran
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(derivative_test ${DERIV_TEST_SRCS})

target_include_directories(derivative_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(derivative_test ${llvm_libs})

add_test(NAME derivative COMMAND derivative_test)

################################################################################
