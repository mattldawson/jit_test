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

namespace jit_test {

class JitDeriv {
public:
  JitDeriv(const int numRxns);
  void Solve();
};
}
#endif // COM_JITDERIV_H

