################################################################################
# Test of chemical forcing (derivative) function

add_subdirectory(Fortran)

################################################################################
# Make the C++ tests
add_executable(derivative_test)

add_dependencies(derivative_test generated-file)

# needed to fine the fc_mangle.h file
target_include_directories(derivative_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_sources(derivative_test 
  PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}/Fortran/preprocessed.F90
    ClassicDeriv.cpp
    CudaJitDeriv.cpp
    JitDeriv.cpp
    JitTest.cpp 
)

target_link_libraries(derivative_test ${llvm_libs} )

add_test(NAME derivative COMMAND derivative_test)

################################################################################
