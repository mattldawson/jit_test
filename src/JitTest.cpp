//===-- src/JitTest.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019 Matthew Dawson
// Licensed under the GNU General Public License version 2 or (at your
// option) any later version. See the file COPYING for details
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Simple program to test JIT compiling for derivative calculations.
///
//===----------------------------------------------------------------------===//

#include "ClassicDeriv.h"
#include "JitDeriv.h"
#include <chrono>
#include <iostream>

#define NUM_RXNS 500

using namespace jit_test;

int main() {
  ClassicDeriv classicDeriv(NUM_RXNS);
  JitDeriv jitDeriv(NUM_RXNS);

  auto start = std::chrono::high_resolution_clock::now();
  classicDeriv.Solve();
  auto stop = std::chrono::high_resolution_clock::now();

  auto classicTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  jitDeriv.Solve();
  stop = std::chrono::high_resolution_clock::now();

  auto jitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << std::endl
            << "Classic: " << classicTime.count()
            << "; JIT: " << jitTime.count() << std::endl;

  return 0;
}
