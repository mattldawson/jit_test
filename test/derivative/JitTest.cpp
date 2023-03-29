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
#include "CudaGeneralDeriv.h"
#include "CudaJitDeriv.h"
#include <openacc.h>
#endif
#include "FortranPreproccessed.h"
#include "deriv_openacc.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <map>
#include <stdlib.h>

#if defined(_OPENACC) || defined(_OPENMP)
#define ACCELERATOR_ENABLED 1
#else
#define ACCELERATOR_ENABLED 0
#endif

#ifndef NUM_REPEAT
#define NUM_REPEAT 10000
#endif

using namespace jit_test;

bool close_enough(const double &first, const double &second,
                  const double tolerance = 0.001) {
  return abs(first - second) < tolerance;
}

using duration = std::chrono::nanoseconds;
int main() {

  std::map<std::string, long long> time_durations;

#ifdef USE_LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
#endif

  ClassicDeriv classicDeriv{};
  double *fClassic;
  double *fPreprocessed;
  fClassic = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                              sizeof(double));
  fPreprocessed = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                                   sizeof(double));

  double *rateConst;
  double *state;
  rateConst = (double *)malloc(classicDeriv.numRxns * classicDeriv.numCell *
                               sizeof(double));
  state = (double *)malloc(classicDeriv.numSpec * classicDeriv.numCell *
                           sizeof(double));

  for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
    for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
      rateConst[i_cell * classicDeriv.numRxns + i_rxn] =
          (rand() % 10000 + 1) / 100.0;
    for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
      state[i_cell * classicDeriv.numSpec + i_spec] = (rand() % 100) / 100.0;
  }

  // Classic Derivative
  auto start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    classicDeriv.Solve(rateConst, state, fClassic);
  auto stop = std::chrono::high_resolution_clock::now();

  time_durations['Classic'] =
      std::chrono::duration_cast<duration>(stop - start).count();

  // CPU Jit Derivative
#ifdef USE_LLVM
  JitDeriv jitDeriv{};
  double *fJit;
  fJit = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                          sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  jitDeriv.DerivCodeGen(classicDeriv);
  stop = std::chrono::high_resolution_clock::now();

  time_durations['CPU JIT Compile Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    jitDeriv.Solve(rateConst, state, fJit);
  stop = std::chrono::high_resolution_clock::now();

  time_durations['CPU JIT Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
#endif

  // Fortran Preprocessed Derivative
  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    preprocessed_solve(rateConst, state, fPreprocessed);
  stop = std::chrono::high_resolution_clock::now();

  time_durations['CPU Preprocessed Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();

  // GPU Jit Derivative
#ifdef USE_GPU
  start = std::chrono::high_resolution_clock::now();
  CudaJitDeriv cudaJitDeriv(classicDeriv, false);
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU JIT Compile Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
  double *fGPUJit;
  fGPUJit = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                             sizeof(double));

  std::chrono::duration<long, std::nano> gpuJitTime =
      std::chrono::nanoseconds::zero();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    gpuJitTime +=
        cudaJitDeriv.Solve(rateConst, state, fGPUJit, classicDeriv.numCell);

  time_durations['GPU JIT Time'] = gpuJitTime.count();

#ifdef USE_COMPILED
  // GPU Jit Derivative (from source)
  double *fGPUJitCompiled;
  fGPUJitCompiled = (double *)calloc(
      classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    cudaJitDeriv.SolveCompiled(rateConst, state, fGPUJitCompiled,
                               classicDeriv.numCell);
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU Preprocessed Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
#endif

  // General GPU derivative
  start = std::chrono::high_resolution_clock::now();
  CudaGeneralDeriv cudaGeneralDeriv(classicDeriv, false);
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU General Compile Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
  double *fGPUGeneral;
  fGPUGeneral = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                                 sizeof(double));

  std::chrono::duration<long, std::nano> gpuGeneralTime =
      std::chrono::nanoseconds::zero();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    gpuGeneralTime +=
        cudaGeneralDeriv.Solve(rateConst, state, fGPUGeneral, classicDeriv);

  time_durations['GPU General Time'] = gpuGeneralTime.count();

#ifdef USE_COMPILED
  // General GPU derivative (from source)
  double *fGPUGeneralCompiled;
  fGPUGeneralCompiled = (double *)calloc(
      classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    cudaGeneralDeriv.SolveCompiled(rateConst, state, fGPUGeneralCompiled,
                                   classicDeriv);
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU General From Source Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
#endif

  // Reordered memory array setup
  double *flippedRateConst;
  double *flippedState;
  double *flippedDeriv;

  flippedRateConst = (double *)malloc(classicDeriv.numRxns *
                                      classicDeriv.numCell * sizeof(double));
  flippedState = (double *)malloc(classicDeriv.numSpec * classicDeriv.numCell *
                                  sizeof(double));
  flippedDeriv = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                                  sizeof(double));

  // Reordered memory GPU JIT derivative
  start = std::chrono::high_resolution_clock::now();
  CudaJitDeriv cudaFlippedJitDeriv(classicDeriv, true);
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU JIT Coallesced Access Compile Time'] =
      std::chrono::duration_cast<duration>(stop - start).count();
  double *fFlippedGPUJit;
  fFlippedGPUJit = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                                    sizeof(double));

  std::chrono::duration<long, std::nano> gpuFlippedJitTime =
      std::chrono::nanoseconds::zero();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep) {
    // Reorder arrays
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
        flippedRateConst[i_cell + classicDeriv.numCell * i_rxn] =
            rateConst[i_cell * classicDeriv.numRxns + i_rxn];
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        flippedState[i_cell + classicDeriv.numCell * i_spec] =
            state[i_cell * classicDeriv.numSpec + i_spec];
    }
    gpuFlippedJitTime += cudaFlippedJitDeriv.Solve(
        flippedRateConst, flippedState, flippedDeriv, classicDeriv.numCell);
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        fFlippedGPUJit[i_cell * classicDeriv.numSpec + i_spec] =
            flippedDeriv[i_cell + classicDeriv.numCell * i_spec];
    }
  }
  time_durations['GPU JIT Coallesced Access'] = gpuFlippedJitTime.count();

#ifdef USE_COMPILED
  // Reordered memory GPU derivative (from source)
  double *fFlippedGPUJitCompiled;
  fFlippedGPUJitCompiled = (double *)calloc(
      classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep) {
    // Reorder arrays
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
        flippedRateConst[i_cell + classicDeriv.numCell * i_rxn] =
            rateConst[i_cell * classicDeriv.numRxns + i_rxn];
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        flippedState[i_cell + classicDeriv.numCell * i_spec] =
            state[i_cell * classicDeriv.numSpec + i_spec];
    }
    cudaFlippedJitDeriv.SolveCompiled(flippedRateConst, flippedState,
                                      flippedDeriv, classicDeriv.numCell);
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        fFlippedGPUJitCompiled[i_cell * classicDeriv.numSpec + i_spec] =
            flippedDeriv[i_cell + classicDeriv.numCell * i_spec];
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU Coallesced Access From Source'] =
      std::chrono::duration_cast<duration>(stop - start).count();
#endif

  // Reordered memory General GPU derivative
  start = std::chrono::high_resolution_clock::now();
  CudaGeneralDeriv cudaFlippedGeneralDeriv(classicDeriv, true);
  stop = std::chrono::high_resolution_clock::now();
  auto gpuFlippedGeneralCompileTime =
      std::chrono::duration_cast<duration>(stop - start);
  double *fFlippedGPUGeneral;
  fFlippedGPUGeneral = (double *)calloc(
      classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  std::chrono::duration<long, std::nano> gpuFlippedGeneralTime =
      std::chrono::nanoseconds::zero();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep) {
    // Reorder arrays
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
        flippedRateConst[i_cell + classicDeriv.numCell * i_rxn] =
            rateConst[i_cell * classicDeriv.numRxns + i_rxn];
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        flippedState[i_cell + classicDeriv.numCell * i_spec] =
            state[i_cell * classicDeriv.numSpec + i_spec];
    }
    gpuFlippedGeneralTime += cudaFlippedGeneralDeriv.Solve(
        flippedRateConst, flippedState, flippedDeriv, classicDeriv);
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        fFlippedGPUGeneral[i_cell * classicDeriv.numSpec + i_spec] =
            flippedDeriv[i_cell + classicDeriv.numCell * i_spec];
    }
  }
  time_durations['GPU General JIT Coallesced'] = gpuFlippedGeneralTime.count();

#ifdef USE_COMPILED
  // Reordered memory General GPU derivative (from source)
  double *fFlippedGPUGeneralCompiled;
  fFlippedGPUGeneralCompiled = (double *)calloc(
      classicDeriv.numSpec * classicDeriv.numCell, sizeof(double));

  start = std::chrono::high_resolution_clock::now();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep) {
    // Reorder arrays
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_rxn = 0; i_rxn < classicDeriv.numRxns; ++i_rxn)
        flippedRateConst[i_cell + classicDeriv.numCell * i_rxn] =
            rateConst[i_cell * classicDeriv.numRxns + i_rxn];
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        flippedState[i_cell + classicDeriv.numCell * i_spec] =
            state[i_cell * classicDeriv.numSpec + i_spec];
    }
    cudaFlippedGeneralDeriv.SolveCompiled(flippedRateConst, flippedState,
                                          flippedDeriv, classicDeriv);
    for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
      for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec)
        fFlippedGPUGeneralCompiled[i_cell * classicDeriv.numSpec + i_spec] =
            flippedDeriv[i_cell + classicDeriv.numCell * i_spec];
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  time_durations['GPU General From Source Coallesced'] =
      std::chrono::duration_cast<duration>(stop - start).count();
#endif

  cudaJitDeriv.OutputCuda("jit.cu");
  cudaFlippedJitDeriv.OutputCuda("jit_flipped.cu");
  cudaGeneralDeriv.OutputCuda("general.cu");
  cudaFlippedGeneralDeriv.OutputCuda("general_flipped.cu");
#endif

#ifdef ACCELERATOR_ENABLED
  double *hderiv_openacc;
  hderiv_openacc = (double *)calloc(classicDeriv.numSpec * classicDeriv.numCell,
                                    sizeof(double));
  std::chrono::duration<long, std::nano> openacc_time =
      std::chrono::nanoseconds::zero();
  for (int i_rep = 0; i_rep < NUM_REPEAT; ++i_rep)
    openacc_time +=
        deriv_openacc(classicDeriv, rateConst, state, hderiv_openacc);
  time_durations['OpenACC'] = openacc_time.count();
#endif

  for (int i_cell = 0; i_cell < classicDeriv.numCell; ++i_cell) {
    for (int i_spec = 0; i_spec < classicDeriv.numSpec; ++i_spec) {
#if 0
      std::cout << std::endl
                << "cell: " << i_cell << " "
                << "fClassic[" << i_spec << "] = " << fClassic[i_spec]
#ifdef USE_LLVM
                << "  fJit[" << i_spec << "] = " << fJit[i_spec]
#endif
                << "  fPreprocessed[" << i_spec << "] = " << fPreprocessed[i_spec]
#ifdef USE_GPU
                << "  fGPUJit[" << i_spec << "] = " << fGPUJit[i_spec]
                << "  fGPUGeneral[" << i_spec << "] = " << fGPUGeneral[i_spec]
                << "  diff[" << i_spec << "] = " << (fGPUGeneral[i_spec] - fClassic[i_spec])
#endif
                << std::endl;
#endif
      int i = i_cell * classicDeriv.numSpec + i_spec;
      assert(close_enough(fClassic[i], fPreprocessed[i]));
#ifdef USE_LLVM
      assert(fClassic[i] == fJit[i]);
#endif
#ifdef USE_GPU
      assert(close_enough(fClassic[i], fGPUJit[i]));
      assert(close_enough(fClassic[i], fFlippedGPUJit[i]));
      assert(close_enough(fClassic[i], fGPUGeneral[i]));
      assert(close_enough(fClassic[i], fFlippedGPUGeneral[i]));
#ifdef USE_COMPILED
      assert(close_enough(fClassic[i], fFlippedGPUGeneralCompiled[i]));
      assert(close_enough(fClassic[i], fFlippedGPUJitCompiled[i]));
      assert(close_enough(fClassic[i], fGPUGeneralCompiled[i]));
      assert(close_enough(fClassic[i], fGPUJitCompiled[i]));
#endif
#endif
    }
  }

  std::cout
      << "Cells, Reactions, Species, Classic, Preprocessed"
#ifdef USE_GPU
#ifdef ACCELERATOR_ENABLED
      << ", OpenACC"
#endif
      << ", GPU JIT, GPU reordered memory JIT, GPU General, GPU reordered "
         "memory general, GPU General (source), GPU reordered (source)"
#endif
#ifdef USE_LLVM
      << ", CPU JIT"
#endif
      << std::endl;

  time_durations["Cells"] = NUM_CELLS;
  time_durations["Reactions"] = NUM_RXNS;
  time_durations["Species"] = NUM_SPEC;

  // the headers
  for (auto it = time_durations.begin(); it != time_durations.end(); ++it) {
    std::cout << it->first;
    if (std::next(it) != time_durations.end()) {
      std::cout << ",";
    }
  }

  // the row values
  std::cout << std::endl;
  for (auto it = time_durations.begin(); it != time_durations.end(); ++it) {
    std::cout << it->second;
    if (std::next(it) != time_durations.end()) {
      std::cout << ",";
    }
  }

  free(rateConst);
  free(state);

  free(fClassic);
  free(fPreprocessed);

#ifdef USE_LLVM
  free(fJit);
#endif

#ifdef USE_GPU
  free(fGPUJit);
  free(fGPUGeneral);
  free(flippedDeriv);
  free(fFlippedGPUJit);
  free(fFlippedGPUGeneral);
  free(flippedRateConst);
  free(flippedState);
#ifdef USE_COMPILED
  free(fGPUJitCompiled);
  free(fGPUGeneralCompiled);
  free(fFlippedGPUJitCompiled);
  free(fFlippedGPUGeneralCompiled);
#endif
#ifdef ACCELERATOR_ENABLED
  free(hderiv_openacc);
#endif
#endif

  return 0;
}
