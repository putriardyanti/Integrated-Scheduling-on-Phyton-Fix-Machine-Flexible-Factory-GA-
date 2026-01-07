"""
- [2025/01/25]    
  See 01-apply-ga-to-flexible-machine-and-facs.drawio
  to understand the encoding_method.   
  This encoding_method is a fusion between 
  [flexible factory]: program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/encoding_twoFacs.py
  and 
  [flexible machine]: gao-2021-implementation/geneticAlgo/encoding.py 
"""

import numpy as np

from numba import njit
from geneticAlgoFlexMacFacs import utils_flexmf


@njit
def roulette_method(one_hot_encoding_arr, fitness_arr, rng=None):
  """For the first k elements, they are not zero
  where k is the last element such that 
  fitness_arr[k+1] == 0.  
  The variable idx_one_hot_encoding also has this same property
  like fitness_arr"""
  r = rng.random()
  fitness_arr_pad = np.zeros(len(fitness_arr) + 1, dtype=np.float32)
  fitness_arr_pad[1:] = fitness_arr.copy()
  fitness_interval = np.cumsum(fitness_arr_pad)

  # to store index when 
  # 1) Psi_1 is 1 for roulette_method when selecting operation, or 
  # 2) idx_Pi_i is 1 for roulette_method when selecting machine
  idx_one_hot_encoding_arr = np.zeros(len(fitness_arr), dtype=np.uint32)  

  flag = 0
  for idx in range(len(one_hot_encoding_arr)):
    if one_hot_encoding_arr[idx] == 1:
      idx_one_hot_encoding_arr[flag] = idx 
      flag += 1

  selected_idx = idx_one_hot_encoding_arr[0]
  for idx, (lower_bound, upper_bound) in enumerate(zip(fitness_interval[:-1], fitness_interval[1:])):
    if lower_bound <= r < upper_bound:
      selected_idx = idx_one_hot_encoding_arr[idx]
      break

  # print(f"  one_hot_encoding_arr")
  # print(one_hot_encoding_arr)
  # print(f"  fitness_arr")
  # print(fitness_arr)
  # print(f"  fitness_interval")
  # print(fitness_interval)
  # print(f"  r: ", r)
  # print(f"  selected_idx: ", selected_idx)
  return selected_idx


@njit
def choose_factory(num_of_factory, rng=None):
  return rng.integers(0, num_of_factory, dtype=np.uint32) + 1   # one-based index

@njit
def encoding_method(inp_Psi_1, inp_Psi_2, inp_Psi_0, inp_R_i, inp_idx_Pi_i, p_ik, 
                    num_of_op, num_of_machine, num_of_factory, inp_T_arr, rng=None, verbose=False):
  chromo_random = np.zeros((3, num_of_op), dtype=np.uint16)
  index_chromo = -1;    # because we use and index enumeration starting from 0
  
  Psi_1 = inp_Psi_1.copy()
  Psi_2 = inp_Psi_2.copy()
  Psi_0 = inp_Psi_0.copy()

  R_i = inp_R_i.copy()
  idx_Pi_i = inp_idx_Pi_i.copy()
  T_arr = inp_T_arr.copy()
  
  # if verbose:
  #   print("Psi_1:", Psi_1)
  #   print("Psi_0:", Psi_0)
  #   print("Psi_2:", Psi_2)

  # for debugging
  selected_idx_arr = np.array([-1], dtype=np.int32)

  while (sum(Psi_1) > 0):
    index_chromo += 1
    # print(f"  index_chromo: ", index_chromo)
    
    # -- sampling of operation
    sum_pl = 0
    for idx, j in enumerate(Psi_1):
      if j:
        sum_pl += R_i[idx]
    # sum_pl = sum(np.array(R_i[idx] for idx, j in enumerate(Psi_1) if j == 1))
    
    # print("sum_pl:", sum_pl)
    
    num_op = sum(Psi_1)
    fitness_op = np.zeros(num_op, dtype=np.float32)

    flag = 0
    for op, stat in enumerate(Psi_1):
      if stat:
        fitness_op[flag] = R_i[op]/sum_pl
        flag += 1
    
    # print("R_i", R_i)
    # print("fitness_op", fitness_op)
    # for debugging
    # break

    v = roulette_method(Psi_1, fitness_op, rng=rng)   # idx of Psi_1
    selected_idx_arr = np.append(selected_idx_arr, [np.int32(v)])
    
    if verbose:
      print("  Psi_1", Psi_1)
      idx_Psi_1 = np.array([idx for idx, cond in enumerate(Psi_1) if cond], dtype=np.int32)
      print("  idx_Psi_1", idx_Psi_1)
      print("  fitness_op", fitness_op)
      print("  v: ",v)

    chromo_random[0, index_chromo] = v

    # -- sampling of machine
    sum_ma = sum(p_ik[v]) 
    num_ma = sum(idx_Pi_i[v])
    fitness_ma = np.zeros(num_ma, dtype=np.float32)

    flag = 0
    for i, stat in enumerate(idx_Pi_i[v]):
      if stat:
        fitness_ma[flag] = 1. - p_ik[v][i]/sum_ma
        flag += 1

    k = roulette_method(idx_Pi_i[v], fitness_ma, rng=rng)
    # k_int = int(k[3:])
    chromo_random[1, index_chromo] = k + 1   # one based index
    chromo_random[2, index_chromo] = choose_factory(num_of_factory, rng=rng)

    # print(f"  k: ", k)
    # print(f"  chromo_random[0]: ", sorted(chromo_random[0]))
    # print(f"  chromo_random: ")
    # print(chromo_random)

    # Move operation Q_v from Psi_1 to Psi_0
    # Psi_1.remove(v)
    # Psi_0.append(v)
    Psi_1[v] = 0
    Psi_0[v] = 1

    if verbose:
      print("  Psi_1: ", Psi_1)
      print("  Psi_0: ", Psi_0)
      idx_Psi_0 = np.array([idx for idx, cond in enumerate(Psi_0) if cond], dtype=np.int32)
      print("  idx_Psi_0", idx_Psi_0)
      print("  Psi_2: ", Psi_2)

    # Get the successor of operation O_v (there is only single successor)
    successor_v = T_arr[v]
    if verbose:
      print("  successor_v: ", successor_v)

    if successor_v == -1:   # we arrive at root node
      break
    else:
      # Get all predecessors of successor_v
      
      predecessor_set = np.array(
        [idx for idx, cond in enumerate(T_arr == successor_v) if cond], dtype=np.int32)
      if verbose:
        print("    predecessor_set: ", predecessor_set)
      
      # Test that the predecessor_set is the subset of Psi_0
      membership = 0
      for predecessor_op in predecessor_set:
        if Psi_0[predecessor_op]:
          membership += 1
      
      # Move operation Q_success_v from Psi_2 to Psi_1
      # We move successor_v to Psi_1 if all predecessors of successor_v are in Psi_0
      if membership == len(predecessor_set):
        # Psi_2.remove(successor_v[0])
        # Psi_1.append(successor_v[0])
        Psi_2[successor_v] = 0
        Psi_1[successor_v] = 1
        
        if verbose:
          print("    Psi_1: ", Psi_1)
          print("    Psi_0: ", Psi_0)
          idx_Psi_0 = np.array([idx for idx, cond in enumerate(Psi_0) if cond], dtype=np.int32)
          print("  idx_Psi_0", idx_Psi_0)
          print("    Psi_2: ", Psi_2)
      
    # for debugging
    # break
    # print(f"selected_idx_arr", selected_idx_arr)
    # print(f"sorted(selected_idx_arr): ", sorted(selected_idx_arr))
    # is_unique = len(np.unique(selected_idx_arr[1:])) == len(selected_idx_arr[1:])
    # print(f"all_unique: ", is_unique)
    # if not is_unique:
    #   break
  return chromo_random


