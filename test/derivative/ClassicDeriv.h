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

#ifndef NUM_RXNS
#define NUM_RXNS 500
#endif
#ifndef NUM_SPEC
#define NUM_SPEC 200
#endif
#ifndef NUM_CELLS
#define NUM_CELLS 10
#endif

namespace jit_test {

class ClassicDeriv {
public:
  ClassicDeriv();
  void Solve(const double *const rateConst, const double *const state, double *const deriv);
  void WritePreprocessedFortran();

  int numRxns = NUM_RXNS;
  int numSpec = NUM_SPEC;
  int numCell = NUM_CELLS;
  int numReact[NUM_RXNS];
  int numProd[NUM_RXNS];
  int reactId[NUM_RXNS][3];
  int prodId[NUM_RXNS][10];
};
}
#endif // COM_CLASSICDERIV_H
