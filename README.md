jit test
========

Testing JIT compiling with LLVM-ORC for math functions.

There are two tests comparing general-purpose math functions
with JIT-compiled equivalents: one for chemical forcing, another
for matrix multiplication.

Both tests use the common header in `include/`.

To run with Docker:

```
docker build -t jit-test .
docker run -it jit-test bash
cd build
make test
```

To run the tests manually in the container:

```
cd /build
./test/derivative/derivative_test
./test/matrix-multiply/matrix_multiply_test
```

On Casper, to run GPU tests:
```
module load cmake/3.22.0 gnu/12.1.0 cuda/11.4.0 nvhpc/22.2
```

You can build and run in an interactive session with:
```
execcasper -A ${MY_PROJECT} --ngpus=1 -l gpu_type=v100 -q gpudev
```
