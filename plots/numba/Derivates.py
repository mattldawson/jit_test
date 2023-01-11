import numpy as np
import numba as nb
from numba import int64, float64

import inspect
import sys

spec = [
  ('n_reactions', int64),
  ('n_species', int64),
  ('rate_constants', float64[:]),
  ('number_of_reactants', int64[:]),
  ('number_of_products', int64[:]),
  ('reactant_id', int64[:, :]),
  ('product_id', int64[:, :]),
]

class ClassicDeriv:

  def __init__(self, n_reactions = 5000, n_species = 2000):
    self.n_reactions = n_reactions
    self.n_species = n_species
    self.rate_constants = np.zeros((self.n_reactions), dtype=np.float64)
    self.number_of_reactants = np.zeros((self.n_reactions), dtype=np.int64)
    self.number_of_products = np.zeros((self.n_reactions), dtype=np.int64)
    self.reactant_id = np.zeros((self.n_reactions, 3), dtype=np.int64)
    self.product_id = np.zeros((self.n_reactions, 10), dtype=np.int64)

  def randomize(self):
    np.random.seed(1856)
    self.rate_constants = (np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 10000 + 1) / 100.0
    self.number_of_reactants = np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 2 + 2
    self.number_of_products = np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 10 + 1

    self.reactant_id[:] = 0
    self.product_id[:] = 0

    for rxn_idx in range(self.n_reactions):
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        self.reactant_id[rxn_idx, reactant_idx] = np.random.randint(0, np.iinfo(np.int64).max) % self.n_species
      for product_idx in range(self.number_of_products[rxn_idx]):
        self.product_id[rxn_idx, product_idx] = np.random.randint(0, np.iinfo(np.int64).max) % self.n_species

  def solve(self, state, deriv):
    deriv[:] = 0
    for rxn_idx in range(self.n_reactions):
      rate = self.rate_constants[rxn_idx]
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        rate *= state[self.reactant_id[rxn_idx, reactant_idx]]
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        deriv[self.reactant_id[rxn_idx, reactant_idx]] -= rate
      for product_idx in range(self.number_of_products[rxn_idx]):
        deriv[self.product_id[rxn_idx, product_idx]] += rate

  def apply(self, parent):
    self.n_reactions = parent.n_reactions
    self.n_species = parent.n_species
    self.rate_constants = np.array(parent.rate_constants)
    self.number_of_reactants = np.array(parent.number_of_reactants)
    self.number_of_products = np.array(parent.number_of_products)
    self.reactant_id = np.array(parent.reactant_id)
    self.product_id = np.array(parent.product_id)



@nb.experimental.jitclass(spec)
class JITDeriv:

  def __init__(self, n_reactions = 5000, n_species = 2000):
    self.n_reactions = n_reactions
    self.n_species = n_species
    self.rate_constants = np.zeros((self.n_reactions), dtype=np.float64)
    self.number_of_reactants = np.zeros((self.n_reactions), dtype=np.int64)
    self.number_of_products = np.zeros((self.n_reactions), dtype=np.int64)
    self.reactant_id = np.zeros((self.n_reactions, 3), dtype=np.int64)
    self.product_id = np.zeros((self.n_reactions, 10), dtype=np.int64)
  
  def randomize(self):
    np.random.seed(1856)
    self.rate_constants = (np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 10000 + 1) / 100.0
    self.number_of_reactants = np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 2 + 2
    self.number_of_products = np.random.randint(0, np.iinfo(np.int64).max, self.n_reactions) % 10 + 1

    self.reactant_id[:] = 0
    self.product_id[:] = 0

    for rxn_idx in range(self.n_reactions):
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        self.reactant_id[rxn_idx, reactant_idx] = np.random.randint(0, np.iinfo(np.int64).max) % self.n_species
      for product_idx in range(self.number_of_products[rxn_idx]):
        self.product_id[rxn_idx, product_idx] = np.random.randint(0, np.iinfo(np.int64).max) % self.n_species

  def solve(self, state, deriv):
    deriv[:] = 0
    for rxn_idx in range(self.n_reactions):
      rate = self.rate_constants[rxn_idx]
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        rate *= state[self.reactant_id[rxn_idx, reactant_idx]]
      for reactant_idx in range(self.number_of_reactants[rxn_idx]):
        deriv[self.reactant_id[rxn_idx, reactant_idx]] -= rate
      for product_idx in range(self.number_of_products[rxn_idx]):
        deriv[self.product_id[rxn_idx, product_idx]] += rate

@nb.jit(nopython=True) 
  # signature_or_function=nb.void(
  #   nb.float64[:],  # state
  #   nb.float64[:],  # deriv
  #   nb.int64,  # n_reactions
  #   nb.int64[:],  # number_of_reactants
  #   nb.float64[:],  # rate_constants
  #   nb.int64[:],  # number_of_products
  #   nb.int64[:, :],  # reactant_id
  #   nb.int64[:, :]), # product_id
  # locals=dict(rate=nb.float64))
def solve(
    state, 
    deriv, 
    n_reactions, 
    number_of_reactants, 
    rate_constants, 
    number_of_products,
    reactant_id, 
    product_id
  ):
  deriv[:] = 0
  for rxn_idx in range(n_reactions):
    rate = rate_constants[rxn_idx]
    for reactant_idx in range(number_of_reactants[rxn_idx]):
      rate *= state[reactant_id[rxn_idx, reactant_idx]]
    for reactant_idx in range(number_of_reactants[rxn_idx]):
      deriv[reactant_id[rxn_idx, reactant_idx]] -= rate
    for product_idx in range(number_of_products[rxn_idx]):
      deriv[product_id[rxn_idx, product_idx]] += rate