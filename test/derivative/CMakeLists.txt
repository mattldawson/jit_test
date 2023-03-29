################################################################################
# Test of chemical forcing (derivative) function

add_subdirectory(Fortran)

################################################################################
# Make the C++ tests
add_executable(derivative_test)

add_dependencies(derivative_test generated-file)

# needed to fine the fc_mangle.h file
target_include_directories(derivative_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(ENABLE_LLVM)
  set(LLVM_SRC JitDeriv.cpp)
endif()

target_sources(derivative_test
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/Fortran/preprocessed.F90
    ClassicDeriv.cpp
    ${LLVM_SRC}
    ${CUDA_SRC}
    JitTest.cpp
    deriv_openacc.cpp
)

target_link_libraries(derivative_test ${llvm_libs} ${NVIDIA_LIBS})

if(ENABLE_GPU)
  target_sources(derivative_test
    PUBLIC
      CudaJitDeriv.cu CudaGeneralDeriv.cu
  )

  set_property(TARGET derivative_test
    PROPERTY CUDA_SEPARABLE_COMPILATION ON
             CUDA_ARCHITECTURES 70
  )

  target_link_libraries(derivative_test OpenACC::OpenACC_CXX)
endif()

add_test(NAME derivative COMMAND derivative_test)

################################################################################
