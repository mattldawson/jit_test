#pragma once

#include <string>
#include "cudaJIT.h"
#include <chrono>

namespace jit_test {

class ClassicDeriv;

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd, bool flipped);
  std::chrono::duration<long, std::nano> Solve(double *rateConst, double *state, double *deriv, int numcell);

private:
  CudaJIT kernelJit;
};

} // namespace jit_test

