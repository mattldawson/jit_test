// Function that returns a JITed version of the matrix multiply function

#ifndef COM_JIT_MAT_MUL_H
#define COM_JIT_MAT_MUL_H

#if __cplusplus
extern "C" {
#endif

typedef void (*MatMulFunc)(int*, int*, int*);

MatMulFunc create_mat_mul_function(const int rowA, const int colA, const int colB);

#if __cplusplus
}
#endif

#endif // COM_JIT_MAT_MUL_H
