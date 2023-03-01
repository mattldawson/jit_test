#include <string>
#include <chrono>
#include "CudaJitDeriv.h"
#include "ClassicDeriv.h"

namespace jit_test {

std::string GenerateCudaKernal(ClassicDeriv cd);
std::string GenerateUnrolledCudaKernal(ClassicDeriv cd);
std::string GenerateMemReorderCudaKernal(ClassicDeriv cd);

CudaJitDeriv::CudaJitDeriv(ClassicDeriv cd)
{
  auto start = std::chrono::high_resolution_clock::now();
  kernelJit(GenerateCudaKernal(cd))
  auto stop = std::chrono::high_resolution_clock::now();
  kernelJitTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  unrolledKernelJit(GenerateUnrolledCudaKernal(cd))
  stop = std::chrono::high_resolution_clock::now();
  unrolledKernelJitTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  start = std::chrono::high_resolution_clock::now();
  MemReorderKernelJit(GenerateMemReorderCudaKernal(cd))
  stop = std::chrono::high_resolution_clock::now();
  MemReorderKernelJitTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
};

void CudaJitDeriv::SolveMemReorder(double *rateConst, double *state, double *deriv, int numcell) {
  CUdeviceptr drateConst, dstate, dderiv;

  // Allocate GPU memory
  CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );
  
  // copy to GPU
  CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, rateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, state, NUM_SPEC * NUM_CELL * sizeof(double)) );
  
  // Call the function
  void *args[] = { &drateConst, &dstate, &dderiv, &numcell };
  
  MemReorderKernelJit.Run(args);
  
  // Get the result
  CUDA_SAFE_CALL( cuMemcpyDtoH(deriv, dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );
  
  CUDA_SAFE_CALL( cuMemFree(drateConst) );
  CUDA_SAFE_CALL( cuMemFree(dstate) );
  CUDA_SAFE_CALL( cuMemFree(dderiv) );
  CUDA_SAFE_CALL( cuMemFree(dnumcell) );
}

void CudaJitDeriv::Solve(double *rateConst, double *state, double *deriv, int numcell) {
  CUdeviceptr drateConst, dstate, dderiv;

  // Allocate GPU memory
  CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );

  // copy to GPU
  CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, rateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, state, NUM_SPEC * NUM_CELL * sizeof(double)) );

  // Call the function
  void *args[] = { &drateConst, &dstate, &dderiv, &numcell };

  KernelJit.Run(args);

  // Get the result
  CUDA_SAFE_CALL( cuMemcpyDtoH(deriv, dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );

  CUDA_SAFE_CALL( cuMemFree(drateConst) );
  CUDA_SAFE_CALL( cuMemFree(dstate) );
  CUDA_SAFE_CALL( cuMemFree(dderiv) );
  CUDA_SAFE_CALL( cuMemFree(dnumcell) );
}

void CudaJitDeriv::SolveUnrolled(double *rateConst, double *state, double *deriv, int numcell) {
  CUdeviceptr drateConst, dstate, dderiv, dnumcell;

  // Allocate GPU memory
  CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemAlloc(&dnumcell, 1 * sizeof(int)) );

  // copy to GPU
  CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, rateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, state, NUM_SPEC * NUM_CELL * sizeof(double)) );
  CUDA_SAFE_CALL( cuMemcpyHtoD(dnumcell, numcell, 1 * sizeof(int)) );

  // Call the function
  void *args[] = { &drateConst, &dstate, &dderiv, &dnumcell };

  unrolledKernelJit.Run(args);

  // Get the result
  CUDA_SAFE_CALL( cuMemcpyDtoH(deriv, dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );

  CUDA_SAFE_CALL( cuMemFree(drateConst) );
  CUDA_SAFE_CALL( cuMemFree(dstate) );
  CUDA_SAFE_CALL( cuMemFree(dderiv) );
  CUDA_SAFE_CALL( cuMemFree(dnumcell) );
}

std::string GenerateMemReorderCudaKernal(ClassicDeriv cd) {
  std::string kernel = "\n\
extern \"C\" __global__                                                      \n\
void solve(double *rateConst, double *state, double *deriv, int numcell)     \n\
                                                                             \n\
{                                                                            \n\
  size_t tid;                                                                \n\
  int i_spec, i_rxn, i_react, i_prod;                                        \n\
  double rate;                                                               \n\
                                                                             \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                               \n\
  if (tid < numcell) {                                                       \n\
     for (i_spec = 0; i_spec < " + std::to_string(cd.numSpec) + "; ++i_spec) \n\
         deriv[i_spec*numcell+tid] = 0.0;                                    \n\
     for (i_rxn = 0; i_rxn < " + std::to_string(cd.numRxns) + "; ++i_rxn) {  \n\
         rate = rateConst[i_rxn*numcell+tid];                                \n\
         for (i_react = 0; i_react < " + cd.numReact[i_rxn] + "; ++i_react)  \n\
             rate *= state[" + std::to_string(cd.reactId[i_rxn][i_react]) + "*numcell+tid]; \n\
         for (i_react = 0; i_react < " + cd.numReact[i_rxn] + "; ++i_react)  \n\
             deriv[" + std::to_string(cd.reactId[i_rxn][i_react]) + "*numcell+tid] -= rate; \n\
         for (i_prod = 0; i_prod < " + cd.numProd[i_rxn] + "; ++i_prod)      \n\
             deriv[" + std::to_string(cd.prodId[i_rxn][i_prod]) + "*numcell+tid] += rate;   \n\
     }                                                                       \n\
  }                                                                          \n\
}                                                                            \n";
  return kernel;
}

std::string GenerateCudaKernal(ClassicDeriv cd) {
  std::string kernel = "\n\
extern \"C\" __global__                                                      \n\
void solve(double *rateConst, double *state, double *deriv, int numcell)     \n\
                                                                             \n\
{                                                                            \n\
  size_t tid;                                                                \n\
  int i_spec, i_rxn, i_react, i_prod;                                        \n\
  double rate;                                                               \n\
                                                                             \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                               \n\
  if (tid < numcell) {                                                       \n\
     for (i_spec = 0; i_spec < " + std::to_string(cd.numSpec) + "; ++i_spec) \n\
         deriv[tid*" + std::to_string(cd.numSpec) + "+i_spec] = 0.0;         \n\
     for (i_rxn = 0; i_rxn < " + std::to_string(cd.numRxns) + "; ++i_rxn) {  \n\
         rate = rateConst[tid*" + std::to_string(cd.numRxns) + "+i_rxn];     \n\
         for (i_react = 0; i_react < " + cd.numReact[i_rxn] + "; ++i_react)  \n\
             rate *= state[tid*" + std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "]; \n\
         for (i_react = 0; i_react < " + cd.numReact[i_rxn] + "; ++i_react)  \n\
             deriv[tid*" + std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "] -= rate; \n\
         for (i_prod = 0; i_prod < " + cd.numProd[i_rxn] + "; ++i_prod)      \n\
             deriv[tid*" + std::to_string(cd.numSpec) + "+" + std::to_string(cd.prodId[i_rxn][i_prod]) + "] += rate;   \n\
     }                                                                       \n\
  }                                                                          \n\
}                                                                            \n";
  return kernel;
}

std::string GenerateUnrolledCudaKernal(ClassicDeriv cd) {

  std::string kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve(double *rateConst, double *state, double *deriv, int numcell)    \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  memset(deriv, 0, N*sizeof(*deriv));                                       \n\
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

  return kernel;
}

} // namespace jit_test

