from Derivates import ClassicDeriv, JITDeriv
from numba import config

import numpy as np
import pandas as pd
import time

def timeit(thing, state, deriv, iter=10_000):
  start = time.time_ns()
  for i in range(iter):
    thing.solve(state, deriv)
  end = time.time_ns()
  return end - start, deriv.copy()

def table_it():
  reactions = []
  species = []
  classic_times = []
  jit_times = []
  jit_compile_times = []

  for nreactions in [10, 100, 500, 1000, 2500, 5000]:
    for nspecies in [10, 100, 500, 1000, 1500, 2000]:
      classic_deriv = ClassicDeriv()
      jit_deriv = JITDeriv(n_reactions = nreactions, n_species = nspecies)

      jit_deriv.randomize()
      classic_deriv.apply(jit_deriv)

      state = (np.random.randint(0, np.iinfo(np.int64).max, classic_deriv.n_species) % 100) / 100.0
      deriv = np.zeros(classic_deriv.n_species)

      classic_time, classic_results = timeit(classic_deriv, state, deriv)
      prime_jit_time, _ = timeit(jit_deriv, state, deriv, iter=1)
      jit_time, jit_results = timeit(jit_deriv, state, deriv)

      reactions.append(nreactions)
      species.append(nspecies)
      classic_times.append(classic_time / 1000) # microseconds
      jit_compile_times.append(prime_jit_time / 1000) # microseconds
      jit_times.append(jit_time / 1000) # microseconds

      assert(np.all(classic_results == jit_results))
  df = pd.DataFrame(
    {
      'reactions': reactions,
      'species': species,
      'Python Classic Time': classic_times,
      'Python JIT Time': jit_times,
      'Python JIT Compile Time': jit_compile_times,
    }
  )
  print(df)
  df.to_csv('../data/python_jit.csv', index=False)

def once():
  classic_deriv = ClassicDeriv()
  jit_deriv = JITDeriv()

  jit_deriv.randomize()
  classic_deriv.apply(jit_deriv)

  state = (np.random.randint(0, np.iinfo(np.int64).max, classic_deriv.n_species) % 100) / 100.0
  deriv = np.zeros(classic_deriv.n_species)

  classic_time, classic_results = timeit(classic_deriv, state, deriv)
  prime_jit_time, _ = timeit(jit_deriv, state, deriv, iter=1)
  jit_time, jit_results = timeit(jit_deriv, state, deriv)

  assert(np.all(classic_results == jit_results))

  print(f'No JIT: {classic_time * 1e6} µs, JIT comp time: {prime_jit_time * 1e6} µs, JIT: {jit_time * 1e6} µs')

def main():
  table_it()

if __name__ == '__main__':
  main()