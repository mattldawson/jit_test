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
