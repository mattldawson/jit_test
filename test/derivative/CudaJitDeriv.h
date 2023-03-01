#pragma once

#include <string>

namespace jit_test {

class ClassicDeriv;

class CudaJitDeriv {
public:
  void Solve(double *rateConst, double *state, double *deriv);
  std::string GenerateCudaKernal(ClassicDeriv cd);

private:
};
}

