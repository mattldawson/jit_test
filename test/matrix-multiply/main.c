#ifdef USE_LLVM
#include "jit_mat_mul.h"
#endif
#include <stdio.h>
#include <stdlib.h>

void mat_mul (int rowA, int colA, int colB, int a[][colA], int b[][colB], int c[][colB]);

int main(){

    int rowA=2, colA=3, rowB=3, colB=2;
    int a[rowA][colA],b[rowB][colB],c[rowA][colB];
    int i,j;

    if(colA != rowB){
      printf("unmatched A_mat's column and B_mat's row\n");
      return 1;
    }

    printf("initialization...\n");
    for(i=0;i<rowA;i++)
       for(j=0;j<colA;j++)
          a[i][j]=i*colA+j;

    for(i=0;i<rowB;i++)
       for(j=0;j<colB;j++)
          b[i][j]=i+j;

    mat_mul(rowA,colA,colB,a,b,c);

//for printing result
    for(i=0;i<rowA;i++){
       for(j=0;j<colB;j++)
          printf("%d\t",c[i][j]);
       printf("\n");
    }

// reset C values
    for(i=0;i<rowA;++i)
      for(j=0;j<colB;++j)
        c[i][j] = 99999;

#ifdef USE_LLVM
    MatMulFunc jit_mat_mul = create_mat_mul_function(rowA, colA, colB);

    (*jit_mat_mul)((int*)a, (int*)b, (int*)c);

//for printing JIT result
    printf("JITed function results:\n");
    for(i=0;i<rowA;i++){
       for(j=0;j<colB;j++)
          printf("%d\t",c[i][j]);
       printf("\n");
    }
#endif

    return 0;
}
