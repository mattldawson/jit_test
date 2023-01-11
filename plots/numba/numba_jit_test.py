from Derivates import ClassicDeriv, JITDeriv, solve
import numba as nb

import numpy as np
import pandas as pd
import time
import os

# doesn't seem to work, but from here
# https://stackoverflow.com/questions/44131691/how-to-clear-cache-or-force-recompilation-in-numba
def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)

def timeit(thing, state, deriv, iter=10_000):
  if isinstance(thing, JITDeriv):
    start = time.time_ns()
    for i in range(iter):
      solve(
        state, deriv,
        n_reactions = thing.n_reactions, 
        number_of_reactants = thing.number_of_reactants, 
        rate_constants = thing.rate_constants, 
        number_of_products = thing.number_of_products,
        reactant_id = thing.reactant_id, 
        product_id = thing.product_id
      )
    end = time.time_ns()
  else:
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

  for nreactions in reversed([10, 100, 500, 1000, 2500, 5000]):
    for nspecies in reversed([10, 100, 500, 1000, 1500, 2000]):
      classic_deriv = ClassicDeriv()
      jit_deriv = JITDeriv(n_reactions = nreactions, n_species = nspecies)

      jit_deriv.randomize()
      classic_deriv.apply(jit_deriv)

      state = (np.random.randint(0, np.iinfo(np.int64).max, classic_deriv.n_species) % 100) / 100.0
      deriv = np.zeros(classic_deriv.n_species)

      classic_time = 0
      # classic_time, classic_results = timeit(classic_deriv, state, deriv)

      with nb.core.event.install_listener('numba:compile', nb.core.event.TimingListener()) as res:
        _ = timeit(jit_deriv, state, deriv, iter=1)
        if res.done:
          # res._duration is in fractional seconds
          # multiple by 1e6 to get to microseconds
          jit_compile_times.append(res._duration * 1e6)
        else:
          jit_compile_times.append(0)

      jit_time, jit_results = timeit(jit_deriv, state, deriv)

      reactions.append(nreactions)
      species.append(nspecies)
      classic_times.append(classic_time / 1000) # microseconds
      jit_times.append(jit_time / 1000) # microseconds

      # assert(np.all(classic_results == jit_results))
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