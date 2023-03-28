#include "ClassicDeriv.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <openacc.h>

// define vector length for OpenACC
#define VECTOR_LENGTH 128

namespace jit_test {

std::chrono::duration<long, std::nano> deriv_openacc(ClassicDeriv cd,
                                                     double *rateConst,
                                                     double *state,
                                                     double *deriv) {
  // Perform the calculation on GPU/device

#ifdef _OPENACC
#pragma acc enter data copyin(                                                 \
    rateConst [0:cd.numRxns * cd.numCell], state [0:cd.numCell * cd.numCell],  \
    cd.numReact [0:cd.numRxns], cd.numProd [0:cd.numRxns],                     \
    cd.reactId [0:cd.numRxns] [0:MAX_REACT],                                   \
    cd.prodId [0:cd.numRxns] [0:MAX_PROD])                                     \
    create(deriv [0:cd.numSpec * cd.numCell])
#elif defined(_OPENMP)
#pragma omp target enter data map(                                             \
    to                                                                         \
    : rateConst [0:cd.numRxns * cd.numCell],                                   \
      state [0:cd.numCell * cd.numCell], cd.numReact [0:cd.numRxns],           \
      cd.numProd [0:cd.numRxns], cd.reactId [0:cd.numRxns] [0:MAX_REACT],      \
      cd.prodId                                                                \
      [0:cd.numRxns] [0:MAX_PROD] map(alloc                                    \
                                      : deriv [0:cd.numSpec * cd.numCell])
#endif

  auto start = std::chrono::high_resolution_clock::now();
#ifdef _OPENACC
#pragma acc parallel default(present)
#pragma loop gang vector vector_length(VECTOR_LENGTH)
#elif defined(_OPENMP)
#pragma omp target teams thread_limit(VECTOR_LENGTH)
#pragma omp distribute parallel for simd
#endif
  for (int i_cell = 0; i_cell < NUM_CELL; ++i_cell) {
    for (int i_spec = 0; i_spec < NUM_SPEC; ++i_spec)
      deriv[i_cell * NUM_SPEC + i_spec] = 0.0;
    for (int i_rxn = 0; i_rxn < NUM_RXN; ++i_rxn) {
      rate = rateConst[i_cell * NUM_RXN + i_rxn];
      for (int i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
        rate *= state[i_cell * NUM_SPEC + hreactId[i_rxn][i_react]];
      for (int i_react = 0; i_react < hnumReact[i_rxn]; ++i_react)
        deriv[i_cell * NUM_SPEC + hreactId[i_rxn][i_react]] -= rate;
      for (int i_prod = 0; i_prod < hnumProd[i_rxn]; ++i_prod)
        deriv[i_cell * NUM_SPEC + hprodId[i_rxn][i_prod]] += rate;
    }
  }

#ifdef _OPENACC
#pragma acc exit data copyout(deriv [0:cd.numSpec * cd.numCell])
#elif defined(_OPENMP)
#pragma omp target exit data map(from : deriv [0:cd.numSpec * cd.numCell])
#endif
  auto stop = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
}

} // namespace jit_test