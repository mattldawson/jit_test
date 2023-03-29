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

set_property(TARGET derivative_test
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(derivative_test ${llvm_libs} ${NVIDIA_LIBS})

if(ENABLE_GPU)
  target_sources(derivative_test
    PUBLIC
      CudaJitDeriv.cu CudaGeneralDeriv.cu
  )

  target_link_libraries(derivative_test OpenACC::OpenACC_CXX)
  target_compile_options(derivative_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_100,code=sm_100>)
endif()

add_test(NAME derivative COMMAND derivative_test)

################################################################################
