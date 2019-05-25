FROM fedora:27

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
    && cd build
