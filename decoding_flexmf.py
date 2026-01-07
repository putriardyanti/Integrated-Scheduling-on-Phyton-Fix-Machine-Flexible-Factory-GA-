"""
Logs
- [2025/01/26]   
  An implementation to decoding for tri-chain.  
  Most of the part in this method is copied from
  `geneticAlgoTwoFactoriesNumba/decoding_twoFacs.py`.    

  For understanding the data structure in some variable in 
  decoding_method, see `01-apply-ga-to-flexible-machine-and-facs.drawio`
"""

import numpy as np

from geneticAlgoFlexMacFacs import utils_flexmf
from numba import njit

@njit
def get_start_time(D_fifj, end_time_all_predecessors, fac, 
                    max_end_time_at_machine_op):
  max_end_time_with_transit = max_end_time_at_machine_op

  # compare to transit time
  for fac_predecessor, \
      end_time_predecessor in zip(end_time_all_predecessors[:, 2], 
                                  end_time_all_predecessors[:, 0]):
    transit_time = end_time_predecessor + D_fifj[fac_predecessor-1, fac-1]

    if transit_time > max_end_time_with_transit:
      max_end_time_with_transit = transit_time

  return max_end_time_with_transit



@njit
def decoding_method(indiv, p_ik, T_arr, num_of_factory, num_of_machine,
                    D_fifj, verbose=False):
  chain_op = indiv[0, :].copy()       # pi_i
  chain_machine = indiv[1, :].copy()   # mu_i 
  chain_factory = indiv[2, :].copy()   # xi_i
  num_of_op = len(chain_op)

  # Auxiliary arrays for idle time
  start_idle_time = np.zeros((num_of_factory, num_of_machine), dtype=np.int32)
  end_idle_time = np.full((num_of_factory, num_of_machine), -1, dtype=np.int32)
  # print(f"  start_idle_time: ", start_idle_time)
  # print(f"  end_idle_time: ", end_idle_time)

  # [period_T_i, idx_machine_i, idx_factory_i]
  # the first row should be a row with all zero elements.
  period_T = np.zeros((num_of_op + 1, 3), dtype=np.int32)


  # [start_time_op_i, machine_op_i, factory_of_op_i]
  starting_time_operation_in_machine = np.full((num_of_op, 3), -1, dtype=np.int32)

  # use can use idx_period_T for debugging to stop at specific idx
  for idx_period_T, (op, mac, fac) in enumerate(zip(chain_op, chain_machine, chain_factory)):
    # print(f"  op, mac, fac", op, mac, fac)

    # Start time and end time of the machine
    start_time = start_idle_time[fac-1][mac-1]
    end_time = end_idle_time[fac-1][mac-1]
    # print(f"  start_time: ", start_time)
    # print(f"  end_time: ", end_time)

    predecessors_of_op = np.array(
      [idx for idx, cond in enumerate(T_arr == op) if cond], dtype=np.int32)
    num_of_predecessors = len(predecessors_of_op)
    
    # Array to check if we need to add travel time between factories
    # [end_time_op_i, idx_machine_i, idx_factory_i]
    end_time_all_predecessors = np.zeros((num_of_predecessors, 3), dtype=np.int32)

    if verbose: 
      print(f"op={op}")
      print(f"predecessors_of_op")
      print(predecessors_of_op)

    if num_of_predecessors > 0:
      # print(f"   number of predecessors={num_of_predecessors}")
      max_end_time_all_predecessors = 0
      machine_of_all_predecessors = np.zeros(num_of_predecessors, dtype=np.int32)
      factory_of_all_predecessors = np.zeros(num_of_predecessors, dtype=np.int32)

      for idx, predecessor in enumerate(predecessors_of_op):
        idx_chain_of_predecessor = np.argwhere(chain_op == predecessor)[0][0]
        # print(f"  chain_op")
        # print(chain_op)
        # print(f"  idx_chain_of_predecessor={idx_chain_of_predecessor}")
        machine_of_predecessor = chain_machine[idx_chain_of_predecessor]
        factory_of_predecessor = chain_factory[idx_chain_of_predecessor]
        machine_of_all_predecessors[idx] = machine_of_predecessor
        factory_of_all_predecessors[idx] = factory_of_predecessor

        if verbose:
          print(f"  mac={machine_of_predecessor}; fac={factory_of_predecessor}; predecessor:{predecessor}")

        end_time_predecessor = (starting_time_operation_in_machine[predecessor, 0] 
          + p_ik[predecessor, machine_of_predecessor-1])

        # Helper variable for adding travel time between factories
        end_time_all_predecessors[idx, 0] = end_time_predecessor
        end_time_all_predecessors[idx, 1] = machine_of_predecessor
        end_time_all_predecessors[idx, 2] = factory_of_predecessor

        if max_end_time_all_predecessors < end_time_predecessor:
          max_end_time_all_predecessors = end_time_predecessor

      # Determine max_end_time_at_machine_op
      if end_time == -1:
        max_end_time_at_machine_op = max_end_time_all_predecessors
      else:
        max_end_time_at_machine_op = max(max_end_time_all_predecessors, end_time)

      # Adding the travel time between factories if necessary
      if verbose:
        print("  end_time_all_predecessors", end_time_all_predecessors)
      
      # -- Get factories that have max_end_time_all_predecessors
      factory_at_max_end_time = np.zeros(num_of_predecessors, dtype=np.int32)
      for idx, end_time_predecessor in enumerate(end_time_all_predecessors[:, 0]):
        if end_time_predecessor == max_end_time_all_predecessors:
          factory_at_max_end_time[idx] = 1
      idx_factory_at_max_end_time = np.zeros(sum(factory_at_max_end_time), dtype=np.int32)
      flag_factory_at_max_end_time = 0
      for idx, cond in enumerate(factory_at_max_end_time):
        if cond:
          idx_factory_at_max_end_time[flag_factory_at_max_end_time] \
            = end_time_all_predecessors[idx, 2]
          flag_factory_at_max_end_time += 1

      if verbose:
        print(f"  idx_factory_at_max_end_time: {idx_factory_at_max_end_time}")


      if np.any(fac != idx_factory_at_max_end_time):
        # If at least one of predecessors of op is coming from another factory
        # different from p, we need to add by D_fifj
       
        start_time_op = get_start_time(D_fifj, end_time_all_predecessors, fac,
                          max_end_time_at_machine_op)
      else:
        start_time_op = max_end_time_at_machine_op
    else:
      # No predecessors
      # start_time_op = start_time if np.isinf(end_time) else end_time
      # print(f"  no predecessors")
      if end_time == -1:
        start_time_op = start_time
      else:
        start_time_op = end_time

    end_time_op = start_time_op + p_ik[op, mac-1]
    
    # we need to offset by one because the first row is reserved for period_T=0
    period_T[idx_period_T+1, 0] = np.int32(end_time_op)
    period_T[idx_period_T+1, 1] = mac
    period_T[idx_period_T+1, 2] = fac

    starting_time_operation_in_machine[op, 0] = np.int32(start_time_op)
    starting_time_operation_in_machine[op, 1] = mac
    starting_time_operation_in_machine[op, 2] = fac

    if verbose:
      print(f"  fac: {fac}; machine: {mac}; ")
      print(f"  start_time_op:" , start_time_op)
      print(f"  end_time_op: ", end_time_op)
      print("  starting_time_operation_in_machine")
      # utils_memetic.pprint_with_indent(starting_time_operation_in_machine, indent=2)
      print(starting_time_operation_in_machine)

    start_idle_time[fac-1][mac-1] = 0
    end_idle_time[fac-1][mac-1] = end_time_op

  # implementing bubble sort
  sorted_period_T = utils_flexmf.array_copy(period_T)
  for i in range(len(sorted_period_T)):
    for j in range(0, len(sorted_period_T) - i - 1):
      # compare based on the second column
      if sorted_period_T[j, 1] > sorted_period_T[j + 1, 1]:
        # Swap rows
        sorted_period_T[j], sorted_period_T[j + 1] \
          = (sorted_period_T[j + 1]).copy(), (sorted_period_T[j]).copy()
      elif sorted_period_T[j, 1] == sorted_period_T[j + 1, 1]:
        # If second column is equal, sort by first column
        if sorted_period_T[j, 0] > sorted_period_T[j + 1, 0]:
            sorted_period_T[j], sorted_period_T[j + 1] \
              = (sorted_period_T[j + 1]).copy(), (sorted_period_T[j]).copy()

    # print for debugging
    # break

  return sorted_period_T, starting_time_operation_in_machine


