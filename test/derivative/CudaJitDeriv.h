#pragma once

#include <string>
#include "cudaJIT.h"
#include "ClassicDeriv.h"

namespace jit_test {

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd, bool flipped);
  void Solve(double *rateConst, double *state, double *deriv, int numcell);
  void SolveCompiled(double *rateConst, double *state, double *deriv, int numcell);
  void OutputCuda(const char *fileName);
private:
  ClassicDeriv classicDeriv;
  bool flipped;
  CudaJIT kernelJit;
};

} // namespace jit_test

