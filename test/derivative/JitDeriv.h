//===-- src/JitDeriv.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019 Matthew Dawson
// Licensed under the GNU General Public License version 2 or (at your
// option) any later version. See the file COPYING for details
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Definition of class describing derivative calculations using JIT
///        compilation.
///
//===----------------------------------------------------------------------===//

#ifndef COM_JITDERIV_H
#define COM_JITDERIV_H

#include "leJIT.h"

namespace jit_test {

class ClassicDeriv;

class JitDeriv {
public:
  JitDeriv();
  void Solve(double *rateConst, double *state, double *deriv);
  void DerivCodeGen(ClassicDeriv cd);

private:
  std::unique_ptr<llvm::orc::leJIT> myJIT;
  double (*funcPtr)(double*, double *, double *);
};
}
#endif // COM_JITDERIV_H

