//===-- src/ClassicDeriv.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019 Matthew Dawson
// Licensed under the GNU General Public License version 2 or (at your
// option) any later version. See the file COPYING for details
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Definition of class describing a classical approach to derivative
///        calculations.
///
//===----------------------------------------------------------------------===//

#ifndef COM_CLASSICDERIV_H
#define COM_CLASSICDERIV_H

namespace jit_test {

class ClassicDeriv {
public:
  ClassicDeriv(const int numRxns);
  void Solve();
};
}
#endif // COM_CLASSICDERIV_H
