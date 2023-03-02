#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include "CudaJitDeriv.h"
#include "ClassicDeriv.h"

namespace jit_test {

std::string GenerateCudaKernel(ClassicDeriv cd, bool flipped);

CudaJitDeriv::CudaJitDeriv(ClassicDeriv cd, bool flipped) :
  classicDeriv(cd), flipped(flipped),
  kernelJit(GenerateCudaKernel(cd, flipped).c_str(), flipped ? "solve_jit_flipped" : "solve_jit" )
{ };

void CudaJitDeriv::Solve(double *rateConst, double *state, double *deriv, int numcell) {
  CUdeviceptr drateConst, dstate, dderiv;

  for (int i = 0; i < NUM_SPEC * NUM_CELLS; ++i) deriv[i] = 0.0;

  // Allocate GPU memory
  CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXNS * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELLS * sizeof(double)) );

  // copy to GPU
  CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, rateConst, NUM_RXNS * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, state, NUM_SPEC * NUM_CELLS * sizeof(double)) );

  // Call the function
  void *args[] = { &drateConst, &dstate, &dderiv, &numcell };

  kernelJit.Run(args);

  // Get the result
  CUDA_SAFE_CALL( cuMemcpyDtoH(deriv, dderiv, NUM_SPEC * NUM_CELLS * sizeof(double)) );

  CUDA_SAFE_CALL( cuMemFree(drateConst) );
  CUDA_SAFE_CALL( cuMemFree(dstate) );
  CUDA_SAFE_CALL( cuMemFree(dderiv) );
}

void CudaJitDeriv::OutputCuda(const char *fileName) {
  std::ofstream outFile;
  outFile.open(fileName);
  outFile << GenerateCudaKernel(this->classicDeriv, this->flipped);
  outFile.close();
}

std::string GenerateCudaKernel(ClassicDeriv cd, bool flipped) {

  std::string kernel;
  if (!flipped) {
    kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve_jit(double *rateConst, double *state, double *deriv, int numcell)    \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  if (tid < numcell) {                                                      \n";

  for (int i_rxn = 0; i_rxn < cd.numRxns; ++i_rxn) {
    kernel += "    rate = rateConst[tid*" + std::to_string(cd.numRxns) + "+" + std::to_string(i_rxn) +"];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    rate *= state[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    deriv[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "] -= rate;\n";

    for (int i_prod = 0; i_prod < cd.numProd[i_rxn]; ++i_prod)
      kernel += "    deriv[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.prodId[i_rxn][i_prod]) + "] += rate;\n";
  }
  kernel += "\n\
  }                                                                         \n\
}                                                                           \n";
  } else {
    kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve_jit_flipped(double *rateConst, double *state, double *deriv, int numcell)    \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  if (tid < numcell) {                                                      \n";

  for (int i_rxn = 0; i_rxn < cd.numRxns; ++i_rxn) {
    kernel += "    rate = rateConst[tid+" + std::to_string(cd.numCell) + "*" + std::to_string(i_rxn) +"];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    rate *= state[tid+"+ std::to_string(cd.numCell) + "*" + std::to_string(cd.reactId[i_rxn][i_react]) + "];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    deriv[tid+"+ std::to_string(cd.numCell) + "*" + std::to_string(cd.reactId[i_rxn][i_react]) + "] -= rate;\n";

    for (int i_prod = 0; i_prod < cd.numProd[i_rxn]; ++i_prod)
      kernel += "    deriv[tid+"+ std::to_string(cd.numCell) + "*" + std::to_string(cd.prodId[i_rxn][i_prod]) + "] += rate;\n";
  }
  kernel += "\n\
  }                                                                         \n\
}                                                                           \n";
  }
  return kernel;
}


} // namespace jit_test

