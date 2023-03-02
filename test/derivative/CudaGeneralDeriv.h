#pragma once

#include <string>
#include "cudaJIT.h"
#include "ClassicDeriv.h"

namespace jit_test {

class CudaGeneralDeriv {
public:
  CudaGeneralDeriv(ClassicDeriv cd, bool flipped);
  void Solve(double *rateConst, double *state, double *deriv, ClassicDeriv cd);
  void OutputCuda(const char *fileName);
private:
  ClassicDeriv classicDeriv;
  bool flipped;
  CudaJIT kernelJit;
};

} // namespace jit_test

