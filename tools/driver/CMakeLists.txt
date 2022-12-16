set(LLVM_LINK_COMPONENTS
  Support
  )

add_jit_test_tool(jit_test
  Driver.cpp
  )

target_link_libraries(jit_test
  PRIVATE
  jitTestBasic
  )
