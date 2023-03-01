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
  CudaJIT kernelJit;
  CudaJIT unrolledKernelJit;
  CudaJIT MemReorderKernelJit;

  std::chrono::duration<double> kernelJitTime;
  std::chrono::duration<double> unrolledKernelJitTime;
  std::chrono::duration<double> MemReorderKernelJitTime;
};
}

