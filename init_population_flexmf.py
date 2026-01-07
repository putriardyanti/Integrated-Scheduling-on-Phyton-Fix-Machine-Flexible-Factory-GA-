# Logs
# - [2025/01/27]   
#   We generate tri-chain in each population. The method for
#   generation of generate_population is copied from
#   geneticAlgoTwoFactoriesNumba/init_population_twoFacs.py

import numpy as np

from geneticAlgoFlexMacFacs import encoding_flexmf, decoding_flexmf
from numba import njit

@njit
def generate_population(N_sample, Psi_1, Psi_2, Psi_0, R_i, idx_Pi_i, p_ik,
                        num_of_op, num_of_factory, num_of_machine, T_arr,
                        D_fifj, rng=None):

  population = np.zeros((N_sample, 3, num_of_op), dtype=np.int32)
  makespan_population = np.zeros(N_sample, dtype=np.int32)
  starting_time_operation_in_machine_population = np.zeros((N_sample, num_of_op, 3), dtype=np.int32)
  
  for pop in range(N_sample):
  # for pop in prange(N_sample):
    # print(f"pop: {pop}")
    indiv = encoding_flexmf.encoding_method(Psi_1, Psi_2, Psi_0, R_i, 
      idx_Pi_i, p_ik, num_of_op, num_of_machine, num_of_factory, T_arr, rng=rng, verbose=False)
    # print(f"indiv", indiv)

    population[pop] = indiv.copy()
    period_T, starting_time_operation = decoding_flexmf.decoding_method(
      indiv, p_ik, T_arr, num_of_factory, num_of_machine, D_fifj, verbose=False)
    # print(f"Psi_1", Psi_1)
    # print(f"Psi_2", Psi_2)
    # print(f"Psi_0", Psi_0)
    # print(f"period_T", period_T.T)
    # print(f"starting_in_operation\n", starting_time_operation.T)
  
    # makespan = 0
    # for fac_vals in period_T[:, 0]:
    #   if makespan < fac_vals:
    #     makespan = fac_vals
    makespan = np.max(period_T[:, 0])

    # for debugging
    # print("p_ik", p_ik.T)
    # print(f"makespan: ", makespan)
    # break

    makespan_population[pop] = makespan
  
    starting_time_operation_in_machine_population[pop] = starting_time_operation.copy()
    # break

  return population, makespan_population, starting_time_operation_in_machine_population