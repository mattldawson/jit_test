#pragma once

#include <string>
#include "cudaJIT.h"
#include <chrono>

namespace jit_test {

class ClassicDeriv;

class CudaGeneralDeriv {
public:
  CudaGeneralDeriv(ClassicDeriv cd, bool flipped);
  std::chrono::duration<long, std::nano> Solve(double *rateConst, double *state, double *deriv, ClassicDeriv cd);

private:
  CudaJIT kernelJit;
};

} // namespace jit_test

