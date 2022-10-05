void mat_mul(int rowA, int colA, int colB, int a[][colA], \
             int b[][colB], int c[][colB]) {

    int i,j,k;

    for(i=0;i<rowA;i++)
       for(j=0;j<colB;j++){
          c[i][j]=0;
          for(k=0;k<colA;k++)
             c[i][j]+=a[i][k]*b[k][j];
       }

}

