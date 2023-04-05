#pragma once

#include <chrono>

namespace jit_test {

class ClassicDeriv;

std::chrono::duration<long, std::nano>
deriv_openacc(ClassicDeriv cd, double *rateConst, double *state, double *deriv);
} // namespace jit_test
