// Pre-generated CUDA kernel source
//

extern "C"
void solve_jit(double *rateConst, double *state, double *deriv, int numcell);
void solve_jit_flipped(double *rateConst, double *state, double *deriv, int numcell);
void solve_general(double *rateConst, double *state, double *deriv, int numcell);
void solve_general_flipped(double *rateConst, double *state, double *deriv, int numcell);
