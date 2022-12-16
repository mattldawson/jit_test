FROM fedora:36

RUN dnf -y update \
    && dnf -y install \
         clang-devel \
         gcc-c++ \
         make \
         llvm-devel \
         cmake \
    && dnf clean all

COPY . /jit_test/

RUN mkdir build \
    && cd build \
    && cmake -G Ninja \
             -D CMAKE_BUILD_TYPE=release \
             /jit_test \
    && ninja \
    && ninja install
