#pragma once

#include <string>
#include "cudaJIT.h"

namespace jit_test {

class ClassicDeriv;

class CudaGeneralDeriv {
public:
  CudaGeneralDeriv(ClassicDeriv cd, bool flipped);
  void Solve(double *rateConst, double *state, double *deriv, ClassicDeriv cd);

private:
  CudaJIT kernelJit;
};

} // namespace jit_test

