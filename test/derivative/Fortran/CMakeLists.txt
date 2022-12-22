################################################################################
# Make C++ generate the fortran code

FortranCInterface_HEADER(
  fc_mangle.h
  SYMBOLS run
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

add_custom_target(
    generated-file
    DEPENDS preprocessed.F90
)

add_library(preprocessed_fortran)
add_dependencies(preprocessed_fortran generated-file)

target_sources(preprocessed_fortran PUBLIC preprocessed.F90)

