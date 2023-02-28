#pragma once

#include <string>
#include "cudaJIT.h"

namespace jit_test {

std::string GenerateCudaKernal(ClassicDeriv cd);

class ClassicDeriv;

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd);
  void Solve(double *rateConst, double *state, double *deriv);
  void SolveUnrolled(double *rateConst, double *state, double *deriv);

private:
  std::unique_ptr<jit_test::cudaJIT> kernelJit;
  std::unique_ptr<jit_test::cudaJIT> unrolledKernelJit;
};
}