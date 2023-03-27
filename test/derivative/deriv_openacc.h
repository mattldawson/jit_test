#pragma once

#include <chrono>

class ClassicDeriv;

std::chrono::duration<long, std::nano> deriv_openacc(ClassicDeriv cd, double* rateConst, double* state, double* deriv);