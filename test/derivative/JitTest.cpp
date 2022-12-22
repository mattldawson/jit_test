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
#include "FortranPreproccessed.h"

#define NUM_REPEAT 10000

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
  double *fPreprocessed;

  state = (double *)malloc(classicDeriv.numSpec * sizeof(double));
  fClassic = (double *)calloc(classicDeriv.numSpec, sizeof(double));
  fJit = (double *)calloc(classicDeriv.numSpec, sizeof(double));
  fPreprocessed = (double *)calloc(classicDeriv.numSpec, sizeof(double));

  for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
    state[i_spec] = (rand() % 100) / 100.0;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    classicDeriv.Solve(state, fClassic);
  auto stop = std::chrono::high_resolution_clock::now();

  auto classicTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  jitDeriv.DerivCodeGen(classicDeriv);

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    jitDeriv.Solve(state, fJit);
  stop = std::chrono::high_resolution_clock::now();

  auto jitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    preprocessed_solve(state, fPreprocessed);
  stop = std::chrono::high_resolution_clock::now();

  auto preprocessedTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec) {
#if 1
    std::cout << std::endl
              << "fClassic[" << i_spec << "] = " << fClassic[i_spec]
              << "  fJit[" << i_spec << "] = " << fJit[i_spec]
              << "  fPreprocessed[" << i_spec << "] = " << fPreprocessed[i_spec];
#endif
    assert(fClassic[i_spec] == fJit[i_spec]);
    assert(fClassic[i_spec] == fPreprocessed[i_spec]);
  }

  std::cout << std::endl
            << std::endl
            << "Classic: " << classicTime.count()
            << "; JIT: " << jitTime.count() 
            << "; Preprocessed: " << preprocessedTime.count() 
            << std::endl
            << "JIT speedup over classic: "
            << ((double)classicTime.count()) / (double)jitTime.count() << std::endl
            << "Preprocessed speedup over classic: "
            << ((double)classicTime.count()) / (double)preprocessedTime.count()
            << std::endl;

  return 0;
}
