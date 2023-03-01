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
  void SolveUnrolled(double *rateConst, double *state, double *deriv, int numcell);
  void SolveMemReorder(double *rateConst, double *state, double *deriv, int numcell);

private:
  std::unique_ptr<jit_test::CudaJIT> kernelJit;
  std::unique_ptr<jit_test::CudaJIT> unrolledKernelJit;
  std::unique_ptr<jit_test::CudaJIT> MemReorderKernelJit;

  std::chrono::duration kernelJitTime;
  std::chrono::duration unrolledKernelJitTime;
  std::chrono::duration MemReorderKernelJitTime;
};
}

