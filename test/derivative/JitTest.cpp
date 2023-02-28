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
#include "CudaJitDeriv.h"
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

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  ClassicDeriv classicDeriv{};
  JitDeriv jitDeriv{};
  CudaJitDeriv cudaJitDeriv{};
  double *rateConst;
  double *state;
  double *fClassic;
  double *fJit;
  double *fPreprocessed;
  double *fGPUJit;

  rateConst = (double *)malloc(classicDeriv.numRxns * classicDeriv.numCell * sizeof(double));
  state = (double *)malloc(classicDeriv.numSpec * classicDeriv.numCell * sizeof(double));
  fClassic = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));
  fJit = (double *)calloc(classicDeriv.numSpec  *classicDeriv.numCell, sizeof(double));
  fPreprocessed = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));
  fGPUJit = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

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

  // Fortran Preprocessed Derivative
  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    preprocessed_solve(rateConst, state, fPreprocessed);
  stop = std::chrono::high_resolution_clock::now();

  auto preprocessedTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  

  // GPU Jit Derivative
  start = std::chrono::high_resolution_clock::now();
  auto kernel_string = cudaJitDeriv.GenerateCudaKernal(classicDeriv);
  std::cout << kernel_string;
  stop = std::chrono::high_resolution_clock::now();

  auto cudaJitCompileTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    cudaJitDeriv.Solve(rateConst, state, fGPUJit);
  stop = std::chrono::high_resolution_clock::now();

  auto gpuJitTime =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

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
      assert(fClassic[i] == fJit[i]);
      assert(close_enough(fClassic[i], fPreprocessed[i]));
      // assert(close_enough(fClassic[i], fGPUJit[i]));
    }
  }

  std::cout << "Classic: " << classicTime.count()
            << "; CPU JIT: " << jitTime.count() 
            << "; Preprocessed: " << preprocessedTime.count() 
            << "; GPU Jit: " << gpuJitTime.count() 
            << std::endl
            << "JIT speedup over classic: "
            << ((double)classicTime.count()) / (double)jitTime.count() << std::endl
            << "Preprocessed speedup over classic: "
            << ((double)classicTime.count()) / (double)preprocessedTime.count() << std::endl
            << "CPU JIT compile time: " << jitCompileTime.count()
            << "GPU JIT compile time: " << cudaJitCompileTime.count()
            << std::endl;

  return 0;
}
