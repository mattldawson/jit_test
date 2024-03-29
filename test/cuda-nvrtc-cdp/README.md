# CUDA NVRTC & CDP samples

# To build and run on CASPER
module load nvhpc/22.2
module load cuda/11.4.0
module load cmake/3.22.0
nvc++ -o saxpy -lnvrtc -lcuda saxpy.cpp
qsub -I -l select=1:ncpus=1:mpiprocs=1:ngpus=1 -l walltime=00:10:00 -l gpu_type=v100 -q casper -A project_id

## Feature list:
 * CUDA Driver API
 * CUDA Runtime Compilation
 * CUDA Dynamic Parallelism
 * saxpy, qsort, mandelbrot samples
 * pageable vs pinned memory sample - transfer times captured with CUDA events

## Requirements
 * CMake
 * NVRTC requires CUDA GPU sm20+
 * CDP requires CUDA GPU sm35+
 * Windows only

<p align="center"><img src="mandelbrot.jpg" width="800" /></p>
<p align="center"><b>Mandelbrot set with CUDA Dynamic Parallelism - The Mariani-Silver Algorithm</b></p>

## References:
 * [NVIDIA CUDA SDK samples](https://github.com/NVIDIA/cuda-samples)
 * [NVRTC saxpy sample - CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/nvrtc/index.html#code-saxpy-cpp)
 * [NVRTC cdp sample - CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/nvrtc/index.html#example-dynamic-parallelism)
 * [Adaptive Parallel Computation with CDP - Parallel Forall](https://devblogs.nvidia.com/parallelforall/introduction-cuda-dynamic-parallelism/)
 * [CUDA Dynamic Parallelism API and Principles - Parallel Forall](https://devblogs.nvidia.com/parallelforall/cuda-dynamic-parallelism-api-principles/)
 * [How to Optimize Data Transfers in CUDA C/C++ - Parallel Forall](https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/)
