#pragma once

#include <string>
#include "cudaJIT.h"

namespace jit_test {

class ClassicDeriv;

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd);
  void Solve(double *rateConst, double *state, double *deriv, int numcell);

private:
  CudaJIT kernelJit;
};

} // namespace jit_test

