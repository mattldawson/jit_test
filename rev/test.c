void getDeriv(double* state, double *deriv) {
  deriv[0] = 0.0;
  deriv[1] = 12.32 * state[2] * state[3];
  deriv[2] = 1.32 * state[1];
  deriv[3] = 3.21 * state[2] * state[0];
}
