#include <string>
#include "CudaJitDeriv.h"
#include "ClassicDeriv.h"

namespace jit_test {

void CudaJitDeriv::Solve(double *rateConst, double *state, double *deriv) {}

std::string CudaJitDeriv::GenerateCudaKernal(ClassicDeriv cd) {

  std::string kernel = "\n\
extern \"C\" __global__                                                     \n\
void solve(double *rateConst, double *state, double *deriv, int numcell)    \n\
{                                                                           \n\
  size_t tid;                                                               \n\
  double rate;                                                              \n\
                                                                            \n\
  tid = blockIdx.x * blockDim.x + threadIdx.x;                              \n\
  memset(deriv, 0, N*sizeof(*deriv));                                       \n\
  if (tid < numcell) {                                                      \n";

  for (int i_rxn = 0; i_rxn < cd.numRxns; ++i_rxn) {
    kernel += "    rate = rateConst[tid*" + std::to_string(cd.numRxns) + "+" + std::to_string(i_rxn) +"];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    rate *= state[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "];\n";

    for (int i_react = 0; i_react < cd.numReact[i_rxn]; ++i_react)
      kernel += "    deriv[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.reactId[i_rxn][i_react]) + "] -= rate;\n";

    for (int i_prod = 0; i_prod < cd.numProd[i_rxn]; ++i_prod)
      kernel += "    deriv[tid*"+ std::to_string(cd.numSpec) + "+" + std::to_string(cd.prodId[i_rxn][i_prod]) + "] += rate;\n";
  }
  kernel += "\n\
  }                                                                         \n\
}                                                                           \n";

  return kernel;
}

} // namespace jit_test