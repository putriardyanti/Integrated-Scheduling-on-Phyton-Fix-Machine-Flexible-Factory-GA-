"""
Logs
- [2025/02/02]   
  A copy of `program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/run_ga_twoFacs.py`
  for flexible machines and factories.ja
"""

import numpy as np

from geneticAlgoFlexMacFacs import (init_population_flexmf,
 local_search_flexmf, decoding_flexmf, selection_flexmf)
from numba import njit

@njit
def run_ga_flexmf(population, makespan_population, 
                    starting_time_operation_in_machine_population, 
                    inp_idx_Pi, inp_p_ik, num_of_op, num_of_factory, num_of_machine,
                    inp_T_arr, P_c, P_m, G_iteration, D_fifj, rng=None, verbose=False):

  dim_of_chain = 3
  N_sample = len(population)
  idx_Pi_i = inp_idx_Pi.copy()
  p_ik = inp_p_ik.copy()
  
  T_arr = inp_T_arr.copy()

  hist_population = np.zeros((G_iteration + 1, N_sample, dim_of_chain, num_of_op), dtype=np.int32)
  hist_population[0] = population.copy()
  hist_new_population = np.zeros((G_iteration + 1, N_sample, dim_of_chain, num_of_op), dtype=np.int32)
  hist_new_population[0] = np.full(population.shape, -1, dtype=np.int32) 
  hist_makespan_population = np.zeros((G_iteration + 1, N_sample), dtype=np.int32)
  hist_makespan_population[0] = makespan_population.copy()
  hist_new_makespan_population = np.zeros((G_iteration + 1, N_sample), dtype=np.int32)
  hist_new_makespan_population[0] = np.full(makespan_population.shape, -1, dtype=np.int32)
  
  # the number 3 in here is not dim_of_chain, but [starting_time_op, machine_op, factory_op]
  hist_starting_time_operation_in_machine_population \
    = np.zeros((G_iteration + 1, N_sample, num_of_op, 3), dtype=np.int32)
  hist_starting_time_operation_in_machine_population[0] \
    = starting_time_operation_in_machine_population.copy()


  for G_i in range(G_iteration):  
    print(f"generation: {G_i}")
    # start_time = time.perf_counter()
    # do selection (tournament strategy and elite retention, including crossover
    # and mutation)

    new_population = selection_flexmf.selection(
      N_sample, population, makespan_population, p_ik, 
      num_of_op, num_of_factory, num_of_machine, T_arr, P_m, P_c, D_fifj, rng=rng)

    print(f"  Selection is complete")
    
    # -- [debugging] test uniqueness of each individual
    for indiv in new_population:
      is_unique = len(indiv[0, :]) == len(np.unique(indiv[0, :]))
      if not is_unique:
        print(indiv)


    # break
    # print(f"  N_sample={N_sample}; len(new_population)={len(new_population)}")

    new_makespan_population = np.zeros(N_sample, dtype=np.int32)
    for idx, indiv in enumerate(new_population):
      # print(f"  --> idx={idx}")
      # print(f"  --> indiv")
      # print(indiv)
      new_period_T, _ = decoding_flexmf.decoding_method(
        indiv, p_ik, T_arr, num_of_factory, num_of_machine, D_fifj)
      # print(f"  --> new_period_T")
      # print(new_period_T)

      # print(f"  --> start calc new_makespan")
      new_makespan = 0
      for period_vals in new_period_T[:, 0]:
        if new_makespan < period_vals:
          new_makespan = period_vals
      # print(f"  --> finish calc new_makespan")
      # print(f"  --> new_makespan={new_makespan}")
      new_makespan_population[idx] = new_makespan
      # print(f"  --> finish updating new_makespan_population")

      # new_makespan_population[idx] = np.max(new_period_T[:, 0])
      


    # print(f"new_population")
    # print(new_population)
    # break

    print(f"  Start local search")
    # -- Local search, always find the best
    best_population = np.zeros((N_sample, dim_of_chain, num_of_op), dtype=np.int32)
    best_makespan = np.zeros(N_sample, dtype=np.int32)

    for idx, chromosome in enumerate(new_population):
      new_chromosome, new_makespan = local_search_flexmf.local_search(
        chromosome, p_ik, idx_Pi_i, T_arr, num_of_factory, num_of_machine, D_fifj)
      best_population[idx] = new_chromosome.copy()
      best_makespan[idx] = new_makespan

    population = best_population.copy()
    makespan_population = best_makespan.copy()
    print(f"  Local search is complete")
    # print(f"population")
    # print(population)
    # print(f"makespan_population")
    # print(makespan_population)
    # display(new_population)
    # break

    # the number 3 in here is not dim_of_chain, but [starting_time_op, machine_op, factory_op]
    starting_time_operation_in_machine_population \
      = np.zeros((N_sample, num_of_op, 3), dtype=np.int32) 
    for idx, indiv in enumerate(population):
      _, starting_time_operation_in_machine = decoding_flexmf.decoding_method(
        indiv, p_ik, T_arr, num_of_factory, num_of_machine, D_fifj)
      starting_time_operation_in_machine_population[idx] \
        = starting_time_operation_in_machine.copy()


    hist_population[G_i+1] = population.copy()
    hist_new_population[G_i+1] = new_population.copy()
    hist_makespan_population[G_i+1] = makespan_population.copy()
    hist_new_makespan_population[G_i+1] = new_makespan_population.copy()
    hist_starting_time_operation_in_machine_population[G_i+1] \
      = starting_time_operation_in_machine_population.copy()

    # print(f"  duration: {time.perf_counter() - start_time:.2f} s")
    
    # for debugging
    # break

  return hist_population, hist_new_population, \
    hist_makespan_population, hist_new_makespan_population, \
    hist_starting_time_operation_in_machine_population