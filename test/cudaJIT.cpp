// Test for CUDA JIT compiler
#include "cudaJIT.h"

using namespace jit_test;

bool close_enough(const double& first, const double& second, const double tolerance = 0.001){
    return abs(first - second) < tolerance;
}

const char *saxpyStr = "                                        \n\
extern \"C\" __global__                                         \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)   \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    out[tid] = a * x[tid] + y[tid];                             \n\
  }                                                             \n\
}                                                               \n";

int main() {

  CudaJIT cudaJit = CudaJIT{ saxpyStr, "saxpy" };

  std::cout << "CUDA code" << std::endl
            << saxpyStr << std::endl << std::endl
            << "PTX" << std::endl
            << cudaJit.Ptx() << std::endl;

  // Generate input for execution, and create output buffers.
  size_t num_items = CUDA_THREADS * CUDA_BLOCKS;
  size_t bufferSize = num_items * sizeof(float);

  float a = 5.1f;
  float *hX = new float[num_items];
  float *hY = new float[num_items];
  float *hOut = new float[num_items];
  for (size_t i = 0; i < num_items; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  CUdeviceptr dX, dY, dOut;
  CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

  // Execute SAXPY
  std::cout << "Running saxpy on " << num_items << " elements ..." << std::endl;

  void *args[] = { &a, &dX, &dY, &dOut, &num_items };
  cudaJit.Run(args);

  // Retrieve and check output
  bool passed = true;
  CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
  for (size_t i = 0; i < num_items; ++i) {
    if ((a * hX[i] + hY[i] - hOut[i])/hOut[i]>1e-6) {
      passed = false;
      break;
    }
  }

  if (passed)
    std::cout << "Passed!" << std::endl;
  else
    std::cout << "Failed!" << std::endl;

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));

    delete[] hX;
    delete[] hY;
    delete[] hOut;

    std::cout << std::endl;
    return 0;
}
