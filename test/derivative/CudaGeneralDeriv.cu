#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include "CudaGeneralDeriv.h"
#include "ClassicDeriv.h"
#ifdef USE_COMPILED
#include "general.cu"
#include "general_flipped.cu"
#endif

namespace jit_test {

std::string GenerateGeneralCudaKernel(ClassicDeriv cd, bool flipped);

CudaGeneralDeriv::CudaGeneralDeriv(ClassicDeriv cd, bool flipped) :
  classicDeriv(cd), flipped(flipped),
  kernelJit(GenerateGeneralCudaKernel(cd, flipped).c_str(), flipped ? "solve_general_flipped" : "solve_general" )
{ };

std::chrono::duration<long, std::nano> CudaGeneralDeriv::Solve(double *rateConst, double *state, double *deriv, ClassicDeriv cd) {
  CUdeviceptr drateConst, dstate, dderiv, dnumReact, dnumProd, dreactId, dprodId;

  for (int i = 0; i < NUM_SPEC * NUM_CELLS; ++i) deriv[i] = 0.0;

  // Save predefined variable for CUDA kernel
  int numcell, numrxn, numspec, maxreact, maxprod;
  numcell  = NUM_CELLS;
  numrxn   = NUM_RXNS;
  numspec  = NUM_SPEC;
  maxreact = MAX_REACT;
  maxprod  = MAX_PROD;

  // Allocate GPU memory
  CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXNS * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dnumReact, NUM_RXNS * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dnumProd, NUM_RXNS * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dreactId, NUM_RXNS * MAX_REACT * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dprodId, NUM_RXNS * MAX_PROD * sizeof(int)) );

  // copy to GPU
  CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, rateConst, NUM_RXNS * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, state, NUM_SPEC * NUM_CELLS * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dnumReact, cd.numReact, NUM_RXNS * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dnumProd, cd.numProd, NUM_RXNS * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dreactId, cd.reactId, NUM_RXNS * MAX_REACT * sizeof(int)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dprodId, cd.prodId, NUM_RXNS * MAX_PROD * sizeof(int)) );

  // Call the function
  void *args[] = { &drateConst, &dstate, &dderiv, &dnumReact, &dnumProd, &dreactId,
                   &dprodId, &numcell, &numrxn, &numspec, &maxreact, &maxprod };

  auto start = std::chrono::high_resolution_clock::now();
  kernelJit.Run(args);
  auto stop = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

  // Get the result
  CUDA_SAFE_CALL( cuMemcpyDtoH(deriv, dderiv, NUM_SPEC * NUM_CELLS * sizeof(double)) );

  CUDA_SAFE_CALL( cuMemFree(drateConst) );
  CUDA_SAFE_CALL( cuMemFree(dstate) );
  CUDA_SAFE_CALL( cuMemFree(dderiv) );
  CUDA_SAFE_CALL( cuMemFree(dnumReact) );
  CUDA_SAFE_CALL( cuMemFree(dnumProd) );
  CUDA_SAFE_CALL( cuMemFree(dreactId) );
  CUDA_SAFE_CALL( cuMemFree(dprodId) );

  return time;
}

std::chrono::duration<long, std::nano> CudaGeneralDeriv::SolveCompiled(double *rateConst, double *state, double *deriv, ClassicDeriv cd) {
  double *drateConst, *dstate, *dderiv;
  int *dnumReact, *dnumProd, *dreactId, *dprodId;

  for (int i = 0; i < NUM_SPEC * NUM_CELLS; ++i) deriv[i] = 0.0;

  // Save predefined variable for CUDA kernel
  int numcell, numrxn, numspec, maxreact, maxprod;
  numcell  = NUM_CELLS;
  numrxn   = NUM_RXNS;
  numspec  = NUM_SPEC;
  maxreact = MAX_REACT;
  maxprod  = MAX_PROD;

  // Allocate GPU memory
  cudaMalloc(&drateConst, NUM_RXNS * NUM_CELLS * sizeof(double));
  cudaMalloc(&dstate, NUM_SPEC * NUM_CELLS * sizeof(double));
  cudaMalloc(&dderiv, NUM_SPEC * NUM_CELLS * sizeof(double));
  cudaMalloc(&dnumReact, NUM_RXNS * sizeof(int));
  cudaMalloc(&dnumProd, NUM_RXNS * sizeof(int));
  cudaMalloc(&dreactId, NUM_RXNS * MAX_REACT * sizeof(int));
  cudaMalloc(&dprodId, NUM_RXNS * MAX_PROD * sizeof(int));

  // copy to GPU
  cudaMemcpy(drateConst, rateConst, NUM_RXNS * NUM_CELLS * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy(dstate, state, NUM_SPEC * NUM_CELLS * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dnumReact, cd.numReact, NUM_RXNS * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(dnumProd, cd.numProd, NUM_RXNS * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(dreactId, cd.reactId, NUM_RXNS * MAX_REACT * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(dprodId, cd.prodId, NUM_RXNS * MAX_PROD * sizeof(int), cudaMemcpyHostToDevice );

  auto start = std::chrono::high_resolution_clock::now();
  // Call the function
#ifdef USE_COMPILED
  if (this->flipped) {
    solve_general_flipped<<<CUDA_BLOCKS,CUDA_THREADS>>>(drateConst, dstate, dderiv,
        dnumReact, dnumProd, dreactId, dprodId, numcell, numrxn, numspec, maxreact, maxprod);
  } else {
    solve_general<<<CUDA_BLOCKS,CUDA_THREADS>>>(drateConst, dstate, dderiv,
        dnumReact, dnumProd, dreactId, dprodId, numcell, numrxn, numspec, maxreact, maxprod);
  }
#endif
  auto stop = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

  // Get the result
  cudaMemcpy(deriv, dderiv, NUM_SPEC * NUM_CELLS * sizeof(double), cudaMemcpyDeviceToHost );

  cudaFree(drateConst);
  cudaFree(dstate);
  cudaFree(dderiv);
  cudaFree(dnumReact);
  cudaFree(dnumProd);
  cudaFree(dreactId);
  cudaFree(dprodId);

  return time;
}

void CudaGeneralDeriv::OutputCuda(const char *fileName) {
  std::ofstream outFile;
  outFile.open(fileName);
  outFile << GenerateGeneralCudaKernel(this->classicDeriv, this->flipped);
  outFile.close();
}

std::string GenerateGeneralCudaKernel(ClassicDeriv cd, bool flipped) {

  std::string kernel;
  if (!flipped) {
    kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve_general(double *rateConst, double *state, double *deriv,                 \n\
           int *numReact, int *numProd, int *reactId, int *prodId,          \n\
           int numcell, int numrxn, int numspec, int maxreact, int maxprod) \n\
                                                                            \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  int i_spec, i_rxn, i_react, i_prod;                                       \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  if (tid < numcell) {                                                      \n\
     for (i_spec = 0; i_spec < numspec; ++i_spec)                           \n\
         deriv[i_spec+numspec*tid] = 0.0;                                   \n\
     for (i_rxn = 0; i_rxn < numrxn; ++i_rxn) {                             \n\
         rate = rateConst[i_rxn+numrxn*tid];                                \n\
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            \n\
             rate *= state[reactId[i_rxn*maxreact+i_react]+numspec*tid];    \n\
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            \n\
             deriv[reactId[i_rxn*maxreact+i_react]+numspec*tid] -= rate;    \n\
         for (i_prod = 0; i_prod < numProd[i_rxn]; ++i_prod)                \n\
             deriv[prodId[i_rxn*maxprod+i_prod]+numspec*tid] += rate;       \n\
     }                                                                      \n\
  }                                                                         \n\
}                                                                           \n";
  } else {
    kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve_general_flipped(double *rateConst, double *state, double *deriv,                 \n\
           int *numReact, int *numProd, int *reactId, int *prodId,          \n\
           int numcell, int numrxn, int numspec, int maxreact, int maxprod) \n\
                                                                            \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  int i_spec, i_rxn, i_react, i_prod;                                       \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  if (tid < numcell) {                                                      \n\
     for (i_spec = 0; i_spec < numspec; ++i_spec)                           \n\
         deriv[i_spec*numcell+tid] = 0.0;                                   \n\
     for (i_rxn = 0; i_rxn < numrxn; ++i_rxn) {                             \n\
         rate = rateConst[i_rxn*numcell+tid];                               \n\
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            \n\
             rate *= state[reactId[i_rxn*maxreact+i_react]*numcell+tid];    \n\
         for (i_react = 0; i_react < numReact[i_rxn]; ++i_react)            \n\
             deriv[reactId[i_rxn*maxreact+i_react]*numcell+tid] -= rate;    \n\
         for (i_prod = 0; i_prod < numProd[i_rxn]; ++i_prod)                \n\
             deriv[prodId[i_rxn*maxprod+i_prod]*numcell+tid] += rate;       \n\
     }                                                                      \n\
  }                                                                         \n\
}                                                                           \n";
  }
  return kernel;
}


} // namespace jit_test

