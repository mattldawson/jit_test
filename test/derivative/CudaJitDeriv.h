#pragma once

#include <string>
#include "cudaJIT.h"
#include <memory>

namespace jit_test {

class ClassicDeriv;

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd);
  void Solve(double *rateConst, double *state, double *deriv, int numcell);

private:
  std::unique_ptr<CudaJIT> kernelJit;
};
}

