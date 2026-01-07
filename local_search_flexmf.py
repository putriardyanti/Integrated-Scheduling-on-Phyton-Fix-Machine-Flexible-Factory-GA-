"""
Logs  
- [2024/02/02]    
  A copy of `program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/local_search_twoFacs.py`
  for flexible machines and factories.

  Our local search perform search for all possible 
  machine and factories. So it is an exhaustive search.

- [2024/04/11]    
  When doing a local search, we need to check first that the machine 
  has non-zero processing time. Previously, we just checked for all machines!
  and that was incorrect.
"""
import numpy as np

from geneticAlgoFlexMacFacs import decoding_flexmf
from numba import njit
  
@njit
def local_search(V, p_ik, idx_Pi_i, T_arr, num_of_factory, num_of_machine, D_fifj):
  period_T, _ = decoding_flexmf.decoding_method(V, p_ik, T_arr, 
    num_of_factory, num_of_machine, D_fifj)
  

  # makespan_V = period_T[-1]
  # This code part is copying from `init_population_memetic.py`
  makespan_V = 0
  for fac_vals in period_T[:, 0]:
    if makespan_V < fac_vals:
      makespan_V = fac_vals
  # makespan_V = np.max(period_T[:, 0])
  # print("makespan_V", makespan_V)

  new_V = V.copy()
  # we need to transpose V using V.T, to unpack for each columns as a pair 
  # of operation and factory
  for idx_col, (op, mac, factory) in enumerate(V.T):
    # All available factories and machines 
    for phi in range(1, num_of_factory+1):
      for mu in range(1, num_of_machine+1):
        if idx_Pi_i[op, mu-1] == 1 and (phi != factory or mu != mac):
          # print(f"idx_Pi_i[op, mu-1] = {idx_Pi_i[op, mu-1]}, phi = {phi}, mac = {mac}, factory = {factory}")
          # store the values mu and phi from the previous iteration
          temp_mu = new_V[1, idx_col]
          temp_phi = new_V[2, idx_col]

          new_V[1, idx_col] = mu
          new_V[2, idx_col] = phi 
          new_period_T, _\
            = decoding_flexmf.decoding_method(
                  new_V, p_ik, T_arr, num_of_factory, 
                  num_of_machine, D_fifj)
          new_makespan = 0
          for fac_vals in new_period_T[:, 0]:
            if new_makespan < fac_vals:
              new_makespan = fac_vals
          # new_makespan = np.max(new_period_T[:, 0])

          if new_makespan < makespan_V:
            makespan_V = new_makespan
          else:
            # See 02-apply-ga-to-flexible-machine-and-facs.drawio
            # restore the previous value
            new_V[1, idx_col] = temp_mu
            new_V[2, idx_col] = temp_phi

  return new_V, makespan_V   # output is new chromosome and its makespan

