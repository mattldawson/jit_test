// NVRTC Compiler for JITing GPU functions
#include <cuda.h>
#include <nvrtc.h>
#include <cstdio>
#include <iostream>
#include <string>

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

#ifndef CUDA_THREADS
#define CUDA_THREADS 128
#endif
#ifndef CUDA_BLOCKS
#define CUDA_BLOCKS  32
#endif

namespace jit_test {

class cudaJIT {
private:
  nvrtcProgram prog;
  CUcontext cuContext;
  CUdevice cuDevice;
  CUmodule cuModule;
  CUfunction cuKernel;
  char *ptx;

public:
  cudaJIT(const char *cudaStr, const char *functionName) {

    std::string fileName = functionName + ".cu";

    // Create an instance of nvrtcProgram with the CUDA code string
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&(this->prog),   // program
        cudaStr,                            // CUDA code string
        fileName,                           // name of CUDA file to generate
        0,                                  // number of headers
        NULL,                               // headers
        NULL));                             // include name

    // Compile the program for compute_35 with fmad disabled
    const char *opts[] =
    {
      "--gpu-architecture=compute_35",
      "--fmad=false"
    };
    nvrtcResult compileResult = nvrtcCompileProgram(this->prog,  // prog
                                                    2,           // number of options
                                                    opts);       // options

    // Export the compilation log and check for successful compile
    this->PrintLog();
    if (compileResult != NVRTC_SUCCESS) exit(1);

    // Obtain PTX from the program
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(this->prog, &ptxSize));
    this->ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(this->prog, this->ptx));

    // Initialize CUDA environment
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&(this->cuDevice), 0));
    CUDA_SAFE_CALL(cuCtxCreate(&(this->cuContext),
                               CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST,
                               this->cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&(this->cuModule), this->ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&(this->cuKernel), cuModule, functionName));

  }

  ~cudaJIT() {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&(this->prog));
    NVRTC_SAFE_CALL(cuModuleUnload(this->cuModule));
    NVRTC_SAFE_CALL(cuCtxDestroy(this->cuContext));
  }

  // Output the compilation log, if one exists
  PrintLog() {
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(this.prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(this.prog, log));
    if (logSize > 1) std::cout << log << '\n';
    delete[] log;
  }

  // Run generated code
  Run(void *args[]) {
    CUDA_SAFE_CALL(
        cuLaunchKernel(this->cuKernel,
                       NUM_BLOCKS, 1, 1,    // grid dim
                       NUM_THREADS, 1, 1,   // block dim
                       0, NULL,             // shared mem and stream
                       args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
  }

}
