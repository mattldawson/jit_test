#pragma once

#include <string>
#include "cudaJIT.h"

namespace jit_test {

class ClassicDeriv;

class CudaJitDeriv {
public:
  CudaJitDeriv(ClassicDeriv cd);
  void Solve(double *rateConst, double *state, double *deriv, int numcell);
  void SolveUnrolled(double *rateConst, double *state, double *deriv, int numcell);
  void SolveMemReorder(double *rateConst, double *state, double *deriv, int numcell);

private:
  std::unique_ptr<jit_test::cudaJIT> kernelJit;
  std::unique_ptr<jit_test::cudaJIT> unrolledKernelJit;
  std::unique_ptr<jit_test::cudaJIT> MemReorderKernelJit;

  std::chrono::duration kernelJitTime;
  std::chrono::duration unrolledKernelJitTime;
  std::chrono::duration MemReorderKernelJitTime;
};
}

