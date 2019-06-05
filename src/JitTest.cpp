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
#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>

using namespace jit_test;

int main() {

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  ClassicDeriv classicDeriv{};
  JitDeriv jitDeriv{};
  double *state;
  double *fClassic;
  double *fJit;

  state = (double *)malloc(classicDeriv.numSpec * sizeof(double));
  fClassic = (double *)calloc(classicDeriv.numSpec, sizeof(double));
  fJit = (double *)calloc(classicDeriv.numSpec, sizeof(double));

  for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
    state[i_spec] = (rand() % 100) / 100.0;

  auto start = std::chrono::high_resolution_clock::now();
  classicDeriv.Solve(state, fClassic);
  auto stop = std::chrono::high_resolution_clock::now();

  auto classicTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  jitDeriv.DerivCodeGen(classicDeriv);

  start = std::chrono::high_resolution_clock::now();
  jitDeriv.Solve(state, fJit);
  stop = std::chrono::high_resolution_clock::now();

  auto jitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec) {
    std::cout << std::endl
              << "fClassic[" << i_spec << "] = " << fClassic[i_spec]
              << "  fJit[" << i_spec << "] = " << fJit[i_spec];
    // assert(fClassic[i_spec] == fJit[i_spec]);
  }

  std::cout << std::endl
            << std::endl
            << "Classic: " << classicTime.count()
            << "; JIT: " << jitTime.count() << std::endl;

  return 0;
}
