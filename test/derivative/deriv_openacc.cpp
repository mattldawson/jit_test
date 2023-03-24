#include <cstdio>
#include <iostream>
#include <cmath>
#include <openacc.h>

// define vector length for OpenACC
#define VECTOR_LENGTH   128 

// define tolerance for verification
#define TOLERANCE       1.e-13

int main(int argc, char **argv)
{
    // Generate input for execution
    int hnumReact[NUM_RXN];
    int hnumProd[NUM_RXN];
    int hreactId[NUM_RXN][MAX_REACT];
    int hprodId[NUM_RXN][MAX_PROD];
    int i_cell, i_rxn, i_react, i_prod, i_spec;

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
            hrateConst[i_cell*NUM_RXN+i_rxn] = (rand() % 10000 + 1) / 100.0;
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec)
            hstate[i_cell*NUM_SPEC+i_spec] = (rand() % 100) / 100.0;
    }

    // Create output buffers.
    double *hderiv, *hderiv_tmp;
    hderiv = (double *)malloc(NUM_SPEC * NUM_CELL * sizeof(double));
    hderiv_tmp = (double *)malloc(NUM_SPEC * NUM_CELL * sizeof(double));  // save the results from GPU 

    // Perform the calculation on CPU/host first
    double rate;
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell){
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec)              
            hderiv[i_cell*NUM_SPEC+i_spec] = 0.0;                      
        for (i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn) {               
            rate = hrateConst[i_cell*NUM_RXN+i_rxn];                  
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                rate *= hstate[i_cell*NUM_SPEC+hreactId[i_rxn][i_react]]; 
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                hderiv[i_cell*NUM_SPEC+hreactId[i_rxn][i_react]] -= rate; 
            for (i_prod = 0; i_prod < hnumProd[i_rxn]; ++i_prod)    
                hderiv[i_cell*NUM_SPEC+hprodId[i_rxn][i_prod]] += rate;   
        }                                                          
    }  

    // Perform the calculation on GPU/device
    #pragma acc enter data copyin(hrateConst[0:NUM_RXN * NUM_CELL], \
                                  hstate[0:NUM_SPEC * NUM_CELL], \
                                  hnumReact[0:NUM_RXN],hnumProd[0:NUM_RXN], \
                                  hreactId[0:NUM_RXN][0:MAX_REACT], \
                                  hprodId[0:NUM_RXN][0:MAX_PROD]) \
                           create(hderiv_tmp[0:NUM_SPEC * NUM_CELL])
    #pragma omp target enter data map(to:hrateConst[0:NUM_RXN * NUM_CELL], \
                                         hstate[0:NUM_SPEC * NUM_CELL], \
                                         hnumReact[0:NUM_RXN],hnumProd[0:NUM_RXN], \
                                         hreactId[0:NUM_RXN][0:MAX_REACT], \
                                         hprodId[0:NUM_RXN][0:MAX_PROD]) \
                                  map(alloc:hderiv_tmp[0:NUM_SPEC * NUM_CELL])

    #pragma acc parallel default(present)
    #pragma loop gang vector vector_length(VECTOR_LENGTH)
    #pragma omp target teams thread_limit(VECTOR_LENGTH)
    #pragma omp distribute parallel for simd 
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell){
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec)
            hderiv_tmp[i_cell*NUM_SPEC+i_spec] = 0.0;
        for (i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn) {
            rate = hrateConst[i_cell*NUM_RXN+i_rxn];
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                rate *= hstate[i_cell*NUM_SPEC+hreactId[i_rxn][i_react]];
            for (i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
                hderiv_tmp[i_cell*NUM_SPEC+hreactId[i_rxn][i_react]] -= rate;
            for (i_prod = 0; i_prod < hnumProd[i_rxn]; ++i_prod)
                hderiv_tmp[i_cell*NUM_SPEC+hprodId[i_rxn][i_prod]] += rate;
        }
    }

    #pragma acc exit data copyout(hderiv_tmp[0:NUM_SPEC * NUM_CELL])
    #pragma omp target exit data map(from:hderiv_tmp[0:NUM_SPEC * NUM_CELL])

    // check output.
    bool passed = true;
    for (i_cell = 0; i_cell < NUM_CELL; ++i_cell) {
        for (i_spec = 0; i_spec < NUM_SPEC; ++i_spec) {
            if ( abs( (hderiv[i_cell*NUM_SPEC+i_spec] - hderiv_tmp[i_cell*NUM_SPEC+i_spec]) / 
                      hderiv[i_cell*NUM_SPEC+i_spec] ) > TOLERANCE ) {
               passed = false;
               break;
            }
        }
    }

    if (passed)
        std::cout << "Passed!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    // Release CPU resources
    delete[] hderiv;
    delete[] hderiv_tmp;
    delete[] hrateConst;
    delete[] hstate; 

    std::cout << std::endl;
    return 0;
}
