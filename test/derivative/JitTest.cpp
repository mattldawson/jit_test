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
#ifdef USE_LLVM
#include "JitDeriv.h"
#endif
#ifdef USE_GPU
#include "CudaJitDeriv.h"
#endif
#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include "FortranPreproccessed.h"

#ifndef NUM_REPEAT
#define NUM_REPEAT 10000
#endif

using namespace jit_test;

bool close_enough(const double& first, const double& second, const double tolerance = 0.001){
  return abs(first - second) < tolerance;
}

int main() {

#ifdef USE_LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
#endif

  ClassicDeriv classicDeriv{};
  double *fClassic;
  double *fPreprocessed;
  fClassic = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));
  fPreprocessed = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  double *rateConst;
  double *state;
  rateConst = (double *)malloc(classicDeriv.numRxns * classicDeriv.numCell * sizeof(double));
  state = (double *)malloc(classicDeriv.numSpec * classicDeriv.numCell * sizeof(double));

  for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
    for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
      rateConst[i_cell*classicDeriv.numRxns+i_rxn] = (rand() % 10000 + 1) / 100.0;
    for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
      state[i_cell*classicDeriv.numSpec+i_spec] = (rand() % 100) / 100.0;
  }

  // Classic Derivative
  auto start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    classicDeriv.Solve(rateConst, state, fClassic);
  auto stop = std::chrono::high_resolution_clock::now();

  auto classicTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // CPU Jit Derivative
#ifdef USE_LLVM
  JitDeriv jitDeriv{};
  double *fJit;
  fJit = (double *)calloc(classicDeriv.numSpec  *classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  jitDeriv.DerivCodeGen(classicDeriv);
  stop = std::chrono::high_resolution_clock::now();

  auto jitCompileTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    jitDeriv.Solve(rateConst, state, fJit);
  stop = std::chrono::high_resolution_clock::now();

  auto jitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
#endif

  // Fortran Preprocessed Derivative
  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    preprocessed_solve(rateConst, state, fPreprocessed);
  stop = std::chrono::high_resolution_clock::now();

  auto preprocessedTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // GPU Jit Derivative
#ifdef USE_GPU
  start = std::chrono::high_resolution_clock::now();
  CudaJitDeriv cudaJitDeriv(classicDeriv);
  stop = std::chrono::high_resolution_clock::now();
  auto gpuJitCompileTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  double *fGPUJit;
  fGPUJit = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    cudaJitDeriv.Solve(rateConst, state, fGPUJit, classicDeriv.numCell);
  stop = std::chrono::high_resolution_clock::now();

  auto gpuJitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
#endif

  for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
    for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec) {
#if 0
      std::cout << std::endl
                << "fClassic[" << i_spec << "] = " << fClassic[i_spec]
                << "  fJit[" << i_spec << "] = " << fJit[i_spec]
                << "  fPreprocessed[" << i_spec << "] = " << fPreprocessed[i_spec]
                << "  fGPUJit[" << i_spec << "] = " << fGPUJit[i_spec]
                << "  diff[" << i_spec << "] = " << (fPreprocessed[i_spec] - fClassic[i_spec]);
#endif
      int i = i_cell * classicDeriv.numSpec + i_spec;
      assert(close_enough(fClassic[i], fPreprocessed[i]));
#ifdef USE_LLVM
      assert(fClassic[i] == fJit[i]);
#endif
#ifdef USE_GPU
      assert(close_enough(fClassic[i], fGPUJit[i]));
#endif
    }
  }

  std::cout << "Classic: " << classicTime.count()
            << "; Preprocessed: " << preprocessedTime.count()
#ifdef USE_LLVM
            << "; CPU JIT: " << jitTime.count()
#endif
#ifdef USE_GPU
            << "; GPU Jit: " << gpuJitTime.count()
#endif
            << std::endl
            << "Preprocessed speedup over classic: "
            << ((double)classicTime.count()) / (double)preprocessedTime.count() << std::endl
#ifdef USE_LLVM
            << "JIT speedup over classic: "
            << ((double)classicTime.count()) / (double)jitTime.count() << std::endl
            << "CPU JIT compile time: " << jitCompileTime.count()
            << std::endl
#endif
#ifdef USE_GPU
            << "GPU JIT speedup over classic: "
            << ((double)classicTime.count()) / (double)gpuJitTime.count()
            << "GPU JIT compile time: " << gpuJitCompileTime.count()
            << std::endl
#endif
            << std::endl;
  return 0;
}
