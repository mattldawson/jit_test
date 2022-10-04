//===-- src/ClassicTest.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019 Matthew Dawson
// Licensed under the GNU General Public License version 2 or (at your
// option) any later version. See the file COPYING for details
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Classic derivative calculation class
///
//===----------------------------------------------------------------------===//

#include "ClassicDeriv.h"
#include <cstdlib>

namespace jit_test {

ClassicDeriv::ClassicDeriv() {
  for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
    this->rateConst[i_rxn] = (rand() % 10000 + 1) / 100.0;
    this->numReact[i_rxn] = rand() % 2 + 2;
    this->numProd[i_rxn] = rand() % 10 + 1;
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
      this->reactId[i_rxn][i_react] = rand() % this->numSpec;
    for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod)
      this->prodId[i_rxn][i_prod] = rand() % this->numSpec;
  }
}
void ClassicDeriv::Solve(const double *const state, double *const deriv) {
  for (int i_spec = 0; i_spec < NUM_SPEC; ++i_spec)
    deriv[i_spec] = 0.0;
  for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
    double rate = this->rateConst[i_rxn];
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
      rate *= state[this->reactId[i_rxn][i_react]];
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
      deriv[this->reactId[i_rxn][i_react]] -= rate;
    for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod)
      deriv[this->prodId[i_rxn][i_prod]] += rate;
  }
}
} // namespace jit_test
