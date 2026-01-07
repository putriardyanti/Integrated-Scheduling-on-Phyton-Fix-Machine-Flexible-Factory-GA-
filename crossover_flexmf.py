"""
Logs 
- [2025/01/28]   
  A copy of `program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/crossover_twoFacs.py`
"""

import numpy as np

from numba import njit

@njit
def get_assembly_op(T_arr):
  max_parent = max(T_arr)

  # array to store the number of predecessors that point to a node
  # from 0 up to max_parent
  num_of_predecessors = np.zeros(max_parent + 1, dtype=np.int32)

  for node in T_arr[1:]:
    num_of_predecessors[node] += 1

  # get the assembly node (a node that has more than 1 predecessor)
  num_of_assembly_nodes = (max_parent + 1) - (
    sum(num_of_predecessors == 0) + sum(num_of_predecessors == 1))
  assembly_ops = np.full(num_of_assembly_nodes, -1, dtype=np.int32)
  
  flag = 0
  for idx, node in enumerate(num_of_predecessors):
    if node > 1:
      assembly_ops[flag] = idx
      flag += 1

  return assembly_ops

@njit
def get_ancestors(T_arr, op):
  # Stack to keep track of nodes to process
  stack = np.array([op], dtype=np.int32)
  # ancestors_of_op = np.array([], dtype=np.int32)
  ancestors_of_op = np.empty(0, dtype=np.int32)

  flag = 0
  while len(stack):
    # Pop an operation from the stack
    current_op = stack[0]
    
    if len(stack[1:]):
      stack = stack[1:].copy()
    else:
      stack = np.empty(0, dtype=np.int32)

    # print(current_op, stack)
    # Find all predecessors of the current operation
    predecessors = np.empty(0, dtype=np.int32)
    for idx, node in enumerate(T_arr):
      if idx != 0 and node == current_op:
        predecessors = np.append(predecessors, [np.int32(idx)])

    # predecessors = np.array([idx for idx, node in enumerate(T_arr) 
    # if idx != 0 and node == current_op], dtype=np.int32)

    # Add the predecessors to the ancestors array
    # print(ancestors_of_op.shape, predecessors.shape)
    ancestors_of_op = np.append(ancestors_of_op, predecessors)

    # Push the predecessors to the stack for further exploration
    stack = np.append(stack, predecessors)

    # flag += 1
    # if flag == 2:
    #   break
  return ancestors_of_op


@njit
def crossover_ops(V1, V2, T_arr, rng=None):
  """Crossover based on assembly operation"""
  assembly_ops = get_assembly_op(T_arr)
  # print(f"assembly_ops: ", assembly_ops)

  # rng = np.random.default_rng()
  # selected_gene = rng.choice([int(ops[1:]) for ops in assembly_ops])
  idx_selected_gene = rng.integers(0, len(assembly_ops), dtype=np.int32)
  selected_gene = assembly_ops[idx_selected_gene]

  descendants_selected_gene = np.sort(get_ancestors(T_arr, selected_gene))
  # print(f"V1:\n", V1)
  # print(f"V2:\n", V2)
  # print(f"selected_gene: ", selected_gene)
  # print(f"descendants_selection_gene\n", descendants_selected_gene)

  V1_prime = np.zeros_like(V1)

  descendant_found_in_V2 = np.zeros((3, len(descendants_selected_gene)), dtype=np.int32)    # this is a stack for simplying the implementation
  flag = 0
  for idx, op in enumerate(V2[0, :]):
    if op in descendants_selected_gene:
      gene = V2[:, idx]
      descendant_found_in_V2[:, flag] = gene
      # descendant_found_in_V2[0, flag] = gene[0]
      # descendant_found_in_V2[1, flag] = gene[1]
      # descendant_found_in_V2[2, flag] = gene[2]
      flag += 1

  # print(f"descendant_found_in_V2:\n", descendant_found_in_V2)

  # -- crossover from V1
  for idx, op in enumerate(V1[0, :]):
    if op not in descendants_selected_gene:
      V1_prime[:, idx] = V1[:, idx]
    else:
      # V1_prime[:, idx] = descendant_found_in_V2.pop(0)
      V1_prime[:, idx] = descendant_found_in_V2[:, 0]
      descendant_found_in_V2 = descendant_found_in_V2[:, 1:].copy()

  return V1_prime, selected_gene



