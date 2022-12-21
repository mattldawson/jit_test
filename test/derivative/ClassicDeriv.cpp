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
#include <string>
#include <fstream>

namespace jit_test {

ClassicDeriv::ClassicDeriv() {
  srand(1856);
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

void ClassicDeriv::WritePreprocessedFortran() {

  std::string module = "";
  // std::string module = 
  //   "module preprocess_test \n"
  //   "\n"
  //   "  implicit none \n"
  //   "  private \n"
  //   "  public :: run \n"
  //   "\n"
  //   "contains\n";

  module += ""
    "\n"
    "  subroutine run(state, deriv)\n"
    "    use iso_c_binding, only: sp=>c_float, dp=>c_double \n"
    "    real(dp), intent(inout) :: state(" + std::to_string(this->numRxns) + ") \n"
    "    real(dp), intent(inout) :: deriv(" + std::to_string(this->numRxns) + ") \n"
    "    \n"
    "    real(dp) :: rateConst(" + std::to_string(this->numRxns) + ") \n"
    "    integer :: numReact(" + std::to_string(this->numRxns) + ") \n"
    "    integer :: numProd(" + std::to_string(this->numRxns) + ") \n"
    "    integer :: reactId(" + std::to_string(this->numRxns) + ",3) \n"
    "    integer :: prodId(" + std::to_string(this->numRxns) + ",10) \n"
    "    real(dp) :: rate \n"
    "    integer :: rxnIdx, reactIdx, prodIdx \n"
    "\n"
    "    state = 0 \n"
    "    deriv = 0 \n"
    "";

  for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
    module +=
      "    rateConst(" + std::to_string(i_rxn+1) + ") = " + std::to_string(this->rateConst[i_rxn]) + "\n"
      "    numReact(" + std::to_string(i_rxn+1) + ") = " + std::to_string(this->numReact[i_rxn]) + "\n"
      "    numProd(" + std::to_string(i_rxn+1) + ") = " + std::to_string(this->numProd[i_rxn]) + "\n";
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react){
      module += "    reactId(" + std::to_string(i_rxn+1) + ", " + std::to_string(i_react+1) + ") = " + std::to_string(this->reactId[i_rxn][i_react]) + " \n";
    }
    for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod){
      module += "    prodId(" + std::to_string(i_rxn+1) + ", " + std::to_string(i_prod+1) + ") = " + std::to_string(this->prodId[i_rxn][i_prod]) + " \n";
    }
  }

  module += 
  "\n"
  "    do rxnIdx = 1, " + std::to_string(this->numRxns) + " \n"
  "      rate = rateConst(rxnIdx) \n"
  "      do reactIdx = 1, numReact(rxnIdx) \n"
  "        rate = rate * state( reactId(rxnIdx, reactIdx ) ) \n"
  "      end do \n"
  "      do reactIdx = 1, numReact(rxnIdx) \n"
  "        deriv( reactId(rxnIdx,reactIdx) ) = deriv( reactId(rxnIdx,reactIdx) ) - rate \n"
  "      end do \n"
  "      do prodIdx = 1, numProd(rxnIdx) \n"
  "        deriv( prodId(rxnIdx,reactIdx) ) = deriv( reactId(rxnIdx,reactIdx) ) + rate \n"
  "      end do \n"
  "    end do \n"
  "  end subroutine run\n";
  
  // module += 
    // "end module preprocess_test\n";
  
  std::ofstream out("preprocessed.F90");
  out << module;
  out.close();
}

} // namespace jit_test
