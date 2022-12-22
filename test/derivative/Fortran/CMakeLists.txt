################################################################################
# Make C++ generate the fortran code

# connect fortran and c++
# https://enccs.github.io/cmake-workshop/cxx-fortran/
# https://github.com/ENCCS/cmake-workshop/blob/e47ac9b5197fc18db513e62ce3f1ac36357d3839/content/code/day-2/25_cxx-fortran/solution/src/math/CMakeLists.txt
FortranCInterface_HEADER(
  fc_mangle.h
  SYMBOLS mod:preprocessed_solve
)

add_executable(generate_fortran GenerateFortran.cpp ../ClassicDeriv.cpp)

target_include_directories(generate_fortran PRIVATE ../)

set_target_properties(generate_fortran
  PROPERTIES 
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
)

add_custom_command(
  POST_BUILD
  OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/preprocessed.F90
  COMMAND
      generate_fortran > preprocessed.F90
  VERBATIM
)

# need this so that the next library doesn't attempt to build until after the binary above
# https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/
add_custom_target(
    generated-file
    DEPENDS preprocessed.F90
)
