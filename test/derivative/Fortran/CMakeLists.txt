################################################################################
# Make C++ generate the fortran code

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
      ${CMAKE_CURRENT_BINARY_DIR}/__header.h # fake, ensures that this command always runs so that the fortran file gets rewritten
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
