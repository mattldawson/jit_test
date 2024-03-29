FROM fedora:36

RUN dnf -y update \
    && dnf -y install \
         clang-devel \
         gcc-c++ \
         gdb \
         gcc-fortran \
         make \
         zlib-devel \
         llvm-devel \
         cmake \
         valgrind \
    && dnf clean all

COPY . /jit_test/

RUN mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=release \
             /jit_test \
    && make
