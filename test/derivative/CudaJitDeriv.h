#pragma once

#include <string>
#include "cudaJIT.h"
#include <chrono>
#include "ClassicDeriv.h"

namespace jit_test {

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd, bool flipped);
  std::chrono::duration<long, std::nano> Solve(double *rateConst, double *state, double *deriv, int numcell);
  std::chrono::duration<long, std::nano> SolveCompiled(double *rateConst, double *state, double *deriv, int numcell);
  void OutputCuda(const char *fileName);
private:
  ClassicDeriv classicDeriv;
  bool flipped;
  CudaJIT kernelJit;
};

} // namespace jit_test

