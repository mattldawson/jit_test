#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <iostream>
#include <cmath>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

// define CUDA thread and block size
// make sure that NUM_THREADS * NUM_BLOCKS > NUM_CELL
#define NUM_THREADS     32
#define NUM_BLOCKS      31250

// define constants for chemical forcing terms
#define NUM_CELL        1000000
#define NUM_RXN         500
#define NUM_SPEC        200
#define MAX_REACT       3      // maximum number of reactants per reaction
#define MAX_PROD        10     // maximum number of products per reaction

// define tolerance for verification
#define TOLERANCE       1.e-6

// function pointer to a CUDA kernel
const char *solveStr = "                                                    \n\
extern \"C\" __global__                                                     \n\
void solve(double *rateConst, double *state, double *deriv,                 \n\
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

int main(int argc, char **argv)
{
    // nvrtc only supports sm20 and above
    // https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

    // Create an instance of nvrtcProgram with the SAXPY code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,           // prog
        solveStr,                           // buffer
        "solve.cu",                         // name
        0,                                  // numHeaders
        NULL,                               // headers
        NULL));                             // includeNames

    // Compile the program for compute_70 (V100) or compute_80 (A100) with fmad disabled.
    const char *opts[] =
    {
        "--gpu-architecture=compute_70",
        "--fmad=false",
        "-lineinfo"
    };
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    3,     // numOptions
                                                    opts); // options

    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL( nvrtcGetProgramLogSize(prog, &logSize) );
    char *log = new char[logSize];
    NVRTC_SAFE_CALL( nvrtcGetProgramLog(prog, log) );
    if (logSize > 1)
        std::cout << log << '\n';
    delete[] log;
    if ( compileResult != NVRTC_SUCCESS ) {
         exit(1);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL( nvrtcGetPTXSize(prog, &ptxSize) );
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL( nvrtcGetPTX(prog, ptx) );

    // Destroy the program.
    NVRTC_SAFE_CALL( nvrtcDestroyProgram(&prog) );

    // Load the generated PTX and get a handle to the Solve kernel.
    CUcontext cuContext;
    CUdevice cuDevice;
    CUmodule cuModule;
    CUfunction cuKernel;

    CUDA_SAFE_CALL( cuInit(0) );
    CUDA_SAFE_CALL( cuDeviceGet(&cuDevice, 0) );
    CUDA_SAFE_CALL( cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cuDevice) );

    CUDA_SAFE_CALL( cuModuleLoadDataEx(&cuModule, ptx, 0, 0, 0) );
    CUDA_SAFE_CALL( cuModuleGetFunction(&cuKernel, cuModule, "solve") );

    // Generate input for execution
    int hnumReact[NUM_RXN];
    int hnumProd[NUM_RXN];
    int hreactId[NUM_RXN][MAX_REACT];
    int hprodId[NUM_RXN][MAX_PROD];
    int i_cell, i_rxn, i_react, i_prod, i_spec;

    // Save predefined variable for CUDA kernel
    int numcell, numrxn, numspec, maxreact, maxprod;
    numcell  = NUM_CELL;
    numrxn   = NUM_RXN;
    numspec  = NUM_SPEC;
    maxreact = MAX_REACT;
    maxprod  = MAX_PROD;

    double *hrateConst;
    double *hstate;
    hrateConst = (double *)malloc(NUM_RXN * NUM_CELL * sizeof(double));
    hstate = (double *)malloc(NUM_SPEC * NUM_CELL * sizeof(double));

    // Randomly initialize the reactant & product number per reaction, 
    // reactant ID and product ID for each reaction
    srand(2023);
    for (i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn) {
        hnumReact[i_rxn] = rand() % (MAX_REACT - 1) + (MAX_REACT - 1);
        hnumProd[i_rxn] = rand() % MAX_PROD + 1;
        for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
            hreactId[i_rxn][i_react] = rand() % NUM_SPEC;
        for (i_prod = 0; i_prod < hnumProd[i_rxn]; ++i_prod)
            hprodId[i_rxn][i_prod] = rand() % NUM_SPEC;
    }

    // Randomly initialize the rateConst and state
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell) {
        for (i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn)
            hrateConst[i_rxn*NUM_CELL+i_cell] = (rand() % 10000 + 1) / 100.0;
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec)
            hstate[i_spec*NUM_CELL+i_cell] = (rand() % 100) / 100.0;
    }

    // Create output buffers.
    double *hderiv, *hderiv_tmp;
    hderiv = (double *)malloc(NUM_SPEC * NUM_CELL * sizeof(double));
    hderiv_tmp = (double *)malloc(NUM_SPEC * NUM_CELL * sizeof(double));  // save the results from GPU 

    // Perform the calculation on CPU/host first
    double rate;
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell){
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec)              
            hderiv[i_spec*NUM_CELL+i_cell] = 0.0;                      
        for (i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn) {               
            rate = hrateConst[i_rxn*NUM_CELL+i_cell];                  
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                rate *= hstate[hreactId[i_rxn][i_react]*NUM_CELL+i_cell]; 
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                hderiv[hreactId[i_rxn][i_react]*NUM_CELL+i_cell] -= rate; 
            for (i_prod = 0; i_prod < hnumProd[i_rxn]; ++i_prod)    
                hderiv[hprodId[i_rxn][i_prod]*NUM_CELL+i_cell] += rate;   
        }                                                          
    }  

    // Allocate GPU memory space for device variables
    CUdeviceptr drateConst, dstate, dderiv, dnumReact, dnumProd, dreactId, dprodId;

    CUDA_SAFE_CALL( cuMemAlloc(&drateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dstate, NUM_SPEC * NUM_CELL * sizeof(double)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dnumReact, NUM_RXN * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dnumProd, NUM_RXN * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dreactId, NUM_RXN * MAX_REACT * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemAlloc(&dprodId, NUM_RXN * MAX_PROD * sizeof(int)) );

    // Copy host data to device
    CUDA_SAFE_CALL( cuMemcpyHtoD(drateConst, hrateConst, NUM_RXN * NUM_CELL * sizeof(double)) );
    CUDA_SAFE_CALL( cuMemcpyHtoD(dstate, hstate, NUM_SPEC * NUM_CELL * sizeof(double)) );
    CUDA_SAFE_CALL( cuMemcpyHtoD(dnumReact, hnumReact, NUM_RXN * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemcpyHtoD(dnumProd, hnumProd, NUM_RXN * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemcpyHtoD(dreactId, hreactId, NUM_RXN * MAX_REACT * sizeof(int)) );
    CUDA_SAFE_CALL( cuMemcpyHtoD(dprodId, hprodId, NUM_RXN * MAX_PROD * sizeof(int)) );

    // Execute Solve kernel on GPU/device
    std::cout << "Running Solve on " << NUM_CELL << " grid cells ..." << std::endl;
    std::cout << "Each grid cell contains " << NUM_SPEC << " species and " << NUM_RXN << " reactions ..." << std::endl;

    void *args[] = { &drateConst, &dstate, &dderiv, &dnumReact, 
                     &dnumProd, &dreactId, &dprodId,
                     &numcell, &numrxn, &numspec, &maxreact, &maxprod 
                   };

    CUDA_SAFE_CALL(
        cuLaunchKernel(cuKernel,
        NUM_BLOCKS, 1, 1,    // grid dim
        NUM_THREADS, 1, 1,   // block dim
        0, NULL,             // shared mem and stream
        args, 0));           // arguments
    CUDA_SAFE_CALL( cuCtxSynchronize() );

    // Retrieve and check output.
    bool passed = true;
    CUDA_SAFE_CALL( cuMemcpyDtoH(hderiv_tmp, dderiv, NUM_SPEC * NUM_CELL * sizeof(double)) );
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell) {
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec) {
            if ( abs( (hderiv[i_spec*NUM_CELL+i_cell] - hderiv_tmp[i_spec*NUM_CELL+i_cell]) / 
                      hderiv[i_spec*NUM_CELL+i_cell] ) > TOLERANCE ) {
               passed = false;
               break;
            }
        }
    }

    if (passed)
        std::cout << "Passed!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    // Release GPU resources
    CUDA_SAFE_CALL( cuMemFree(drateConst) );
    CUDA_SAFE_CALL( cuMemFree(dstate) );
    CUDA_SAFE_CALL( cuMemFree(dderiv) );
    CUDA_SAFE_CALL( cuMemFree(dnumReact) );
    CUDA_SAFE_CALL( cuMemFree(dnumProd) );
    CUDA_SAFE_CALL( cuMemFree(dreactId) );
    CUDA_SAFE_CALL( cuMemFree(dprodId) );
    CUDA_SAFE_CALL( cuModuleUnload(cuModule) );
    CUDA_SAFE_CALL( cuCtxDestroy(cuContext) );

    // Release CPU resources
    delete[] hderiv;
    delete[] hderiv_tmp;
    delete[] hrateConst;
    delete[] hstate; 

    std::cout << std::endl;
    return 0;
}
