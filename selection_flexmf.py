"""  
Logs  
- [2025/02/02] 
  This implementation of selection is a copy of 
  `program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/selection_twoFacs.py`

- [2025/02/16]    
  Improve shuffling procedure for population and makespan_population
"""

import numpy as np

from geneticAlgoFlexMacFacs import crossover_flexmf, mutation_flexmf, decoding_flexmf
from numba import njit

@njit
def selection(N_sample, population, makespan_population, p_ik, 
              num_of_op, num_of_factory, num_of_machine, T_arr, P_m, P_c, 
              D_fifj, rng=None):
  
  dim_of_chain = 3
  number_of_population_enter_crossover_pool = np.int32(P_c*N_sample)

  idx_population = np.arange(N_sample, dtype=np.int32)
  rng.shuffle(idx_population)
  # rng.shuffle(population)
  population = population[idx_population]
  makespan_population = makespan_population[idx_population]

  population_not_enter_crossover \
    = population[:N_sample-number_of_population_enter_crossover_pool].copy()

  new_population = population_not_enter_crossover.copy()

  threshold_num_mutated_gene = np.int32(num_of_op*P_m)
  
  while True:
    # -- Tournament selection
    # rng = np.random.default_rng()

    # take 4 individuals from population randomly
    # idx_choices = rng.choice(range(N_sample), 4)
    idx_choices = rng.integers(N_sample-number_of_population_enter_crossover_pool, N_sample, 4, dtype=np.int32)
    makespan_choices = np.array([makespan_population[idx] for idx in idx_choices], dtype=np.int32)
    arg_makespan_sorted = np.argsort(makespan_choices)
    makespan_choices_sorted = (makespan_choices[arg_makespan_sorted]).copy()
    idx_choices_sorted = (idx_choices[arg_makespan_sorted]).copy()
    
    # idx_makespan_choices_sorted

    # take 2 best samples based on their makespan
    idx_choices_best = idx_choices_sorted[:2]
    # idx_choices_best

    # -- Elite retention strategy
    # Crossover
    idx_V1 = idx_choices_best[0]
    idx_V2 = idx_choices_best[1]
    V1_prime, selected_gene_V1 = crossover_flexmf.crossover_ops(
      population[idx_V1], population[idx_V2], T_arr, rng=rng)
    V2_prime, selected_gene_V2 = crossover_flexmf.crossover_ops(
      population[idx_V2], population[idx_V1], T_arr, rng=rng)
    # print(f"  --> Complete crossover")

    # -- [debugging] test uniqueness of each individual
    # is_unique = len(V1_prime[0, :]) == len(np.unique(V1_prime[0, :]))
    # if not is_unique:
    #   print(f"V1_prime")
    #   print(V1_prime)
    # is_unique = len(V2_prime[0, :]) == len(np.unique(V2_prime[0, :]))
    # if not is_unique:
    #   print(f"V2_prime")
    #   print(V2_prime)


    # Mutation
    V1_double_prime, num_mutated_gene_1 = mutation_flexmf.mutation_ops(
      V1_prime, T_arr, rng=rng)  # this is the offspring
    V2_double_prime, num_mutated_gene_2 = mutation_flexmf.mutation_ops(
      V2_prime, T_arr, rng=rng)

    # -- [debugging] test uniqueness of each individual
    is_unique = len(V1_double_prime[0, :]) == len(np.unique(V1_double_prime[0, :]))
    if not is_unique:
      print(f"V1_prime")
      print(V1_prime)
      print(f"V1_double_prime")
      print(V1_double_prime)
    is_unique = len(V2_double_prime[0, :]) == len(np.unique(V2_double_prime[0, :]))
    if not is_unique:
      print(f"V2_prime")
      print(V2_prime)
      print(f"V2_double_prime")
      print(V2_double_prime)
    
    
    if num_mutated_gene_1 <= threshold_num_mutated_gene:
      # If makespan of V1_double_prime is not better than V1, replace it with V1
      period_T_V1_double_prime, _ = decoding_flexmf.decoding_method(
        V1_double_prime, p_ik, T_arr, num_of_factory, num_of_machine, D_fifj)

      # makespan_V1_double_prime = period_T_V1_double_prime[-1]
      # This code part is copying from `init_population_twoFacs.py`
      # makespan_V1_double_prime = 0
      # for fac_vals in period_T_V1_double_prime[:, 0]:
      #   if makespan_V1_double_prime < fac_vals:
      #     makespan_V1_double_prime = fac_vals
      makespan_V1_double_prime = np.max(period_T_V1_double_prime[:, 0])

      if makespan_choices_sorted[0] < makespan_V1_double_prime:
        best_offspring = population[idx_V1]
      else:
        best_offspring = V1_double_prime
      
      # new_population = np.append(new_population, np.array([best_offspring]), axis=0)
      temp = new_population.copy()
      new_population = np.zeros((len(temp)+1, dim_of_chain, num_of_op), dtype=np.int32)
      new_population[:-1, :, :] = temp.copy()
      new_population[-1, :, :] = best_offspring.copy()

    else:
      # new_population = np.append(new_population, np.array([V1_double_prime]), axis=0)
      temp = new_population.copy()
      new_population = np.zeros((len(temp)+1, dim_of_chain, num_of_op), dtype=np.int32)
      new_population[:-1, :, :] = temp.copy()
      new_population[-1, :, :] = V1_double_prime.copy()
    
    # # Do the same for V2_double_prime
    # if num_mutated_gene_2 <= threshold_num_mutated_gene:
    #   period_T_V2_double_prime, _ = decoding_twoFacs.decoding_method(
    #     V2_double_prime[0], V2_double_prime[1], p_ik, T_arr, 
    #     num_of_factory, num_of_machine, D_f1f2=D_f1f2)

    #   # makespan_V2_double_prime = period_T_V2_double_prime[-1]
    #   # This code part is copying from `init_population_memetic.py`
    #   makespan_V2_double_prime = 0
    #   for fac_vals in period_T_V2_double_prime[:, 0]:
    #     if makespan_V2_double_prime < fac_vals:
    #       makespan_V2_double_prime = fac_vals

    #   if makespan_choices_sorted[1] < makespan_V2_double_prime:
    #     best_offspring = population[idx_V2]
    #   else:
    #     best_offspring = V2_double_prime
    #   new_population = np.append(new_population, np.array([best_offspring]), axis=0)
    # else:
    #   new_population = np.append(new_population, np.array([V2_double_prime]), axis=0)


    # print(f"  -- create new individual; len(new_population): {len(new_population)}")
    if len(new_population) > N_sample:
      break

  return new_population[:N_sample] 
  
