################################################################################
# JIT tests

add_subdirectory(derivative)
add_subdirectory(matrix-multiply)

################################################################################
# JIT tests

if (ENABLE_GPU)
  add_executable(cuda_JIT_test)
  target_include_directories(cuda_JIT_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_sources(cuda_JIT_test
    PUBLIC
      cudaJIT.cpp
  )
target_link_libraries(cuda_JIT_test ${NVIDIA_LIBS})
  add_test(NAME cuda_JIT COMMAND cuda_JIT_test)
endif()

################################################################################
