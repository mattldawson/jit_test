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
#include <iostream>

namespace jit_test {

ClassicDeriv::ClassicDeriv() {
  srand(1856);
  for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
    this->numReact[i_rxn] = rand() % (MAX_REACT - 1) + 2;
    this->numProd[i_rxn] = rand() % MAX_PROD + 1;
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
      this->reactId[i_rxn][i_react] = rand() % this->numSpec;
    for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod)
      this->prodId[i_rxn][i_prod] = rand() % this->numSpec;
  }
}
void ClassicDeriv::Solve(const double *const rateConst, const double *const state, double *const deriv) {
  for (int i_cell = 0; i_cell < this->numCell; ++i_cell) {
    for (int i_spec = 0; i_spec < this->numSpec; ++i_spec)
      deriv[i_cell*this->numSpec+i_spec] = 0.0;
    for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
      double rate = rateConst[i_cell*this->numRxns+i_rxn];
      for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
        rate *= state[i_cell*this->numSpec+this->reactId[i_rxn][i_react]];
      for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
        deriv[i_cell*this->numSpec+this->reactId[i_rxn][i_react]] -= rate;
      for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod)
        deriv[i_cell*this->numSpec+this->prodId[i_rxn][i_prod]] += rate;
    }
  }
}

void ClassicDeriv::WritePreprocessedFortran() {

  std::string module =
    "module mod\n"
    "use iso_c_binding \n"
    "use iso_c_binding, only: sp=>c_float, dp=>c_double \n"
    "implicit none\n"
    "private \n"
    "public :: preprocessed_solve\n"
    "contains\n"
    "  subroutine preprocessed_solve(rate_const, state, deriv) bind(c)\n"
    "    use iso_c_binding \n"
    "    real(dp), intent(in) :: rate_const(" + std::to_string(this->numRxns *
                                                               this->numCell)+") \n"
    "    real(dp), intent(in) :: state(" + std::to_string(this->numSpec *
                                                          this->numCell)+") \n"
    "    real(dp), intent(inout) :: deriv(" + std::to_string(this->numSpec *
                                                             this->numCell) + ") \n"
    "\n"
    "    real(dp) :: rate \n"
    "    integer :: i_cell\n"
    "\n"
    "    deriv = 0 \n"
    "\n"
    "    do i_cell = 0, " + std::to_string(this->numCell-1) + "\n"
    "";


  for (int i_rxn = 0; i_rxn < this->numRxns; ++i_rxn) {
    module += "      rate = rate_const(" + std::to_string(i_rxn+1) + " + " +
                                           std::to_string(this->numRxns) + " * i_cell)\n";
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
      module += "      rate = rate * state( " + std::to_string(this->reactId[i_rxn][i_react] + 1) + " + " +
                                                std::to_string(this->numSpec) + " * i_cell) \n";
    for (int i_react = 0; i_react < this->numReact[i_rxn]; ++i_react)
    {
      module += 
      "      deriv( " + std::to_string(this->reactId[i_rxn][i_react] + 1) + " + " +
                        std::to_string(this->numSpec) + " * i_cell ) = "
          "deriv( " + std::to_string(this->reactId[i_rxn][i_react] + 1) + " + " +
                      std::to_string(this->numSpec) + " * i_cell ) - rate \n";
    }
    for (int i_prod = 0; i_prod < this->numProd[i_rxn]; ++i_prod)
    {
      module += 
      "      deriv( " + std::to_string(this->prodId[i_rxn][i_prod] + 1) + " + " +
                        std::to_string(this->numSpec) + " * i_cell ) = "
          "deriv( " + std::to_string(this->prodId[i_rxn][i_prod] + 1) + " + " +
                      std::to_string(this->numSpec) + " * i_cell ) + rate \n";
    }
  }

  module += 
    "    end do\n"
    "\n"
    "  end subroutine preprocessed_solve\n"
    "end module mod\n";
  
  std::cout << module << std::endl;
}

} // namespace jit_test
