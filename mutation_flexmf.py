"""
Logs   
- [2025/01/28]   
  A copy of `program-gao-2021-ga-numba/geneticAlgoTwoFactoriesNumba/mutation_twoFacs.py`

- [2025/02/17]    
  Fix all_stacks creation. In the beginning, we falsely stacked 
  the multiple stack by putting a wrong index of the stack 
  in its `offset_all_stacks`. 

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
    # -- do not use empty for initialization of np.array
    #    if you apply Numba to your program
    predecessors = np.zeros(1, dtype=np.int32)
    for idx, node in enumerate(T_arr):
      if idx != 0 and node == current_op:
        predecessors = np.append(predecessors, [np.int32(idx)])
    predecessors = predecessors[1:]

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
def create_stack_all_children(T_arr, child_nodes):
  flat_stacks = np.empty(0, dtype=np.int32)
  offset_flat_stacks = np.zeros(len(child_nodes), dtype=np.int32)
  length_flat_stacks = np.zeros(len(child_nodes), dtype=np.int32)

  for idx, child in enumerate(child_nodes[:-1]):
    flat_stacks = np.append(flat_stacks, [child])
    ancestors = get_ancestors(T_arr, child)
    flat_stacks = np.append(flat_stacks, ancestors)
    # + 1 because we also include the child node
    offset_flat_stacks[idx + 1] = offset_flat_stacks[idx] + (len(ancestors) + 1)
    length_flat_stacks[idx] = len(ancestors) + 1

  # last stack
  flat_stacks = np.append(flat_stacks, [child_nodes[-1]])
  ancestors = get_ancestors(T_arr, child_nodes[-1])
  flat_stacks = np.append(flat_stacks, ancestors)
  length_flat_stacks[-1] = len(ancestors) + 1

  return flat_stacks, offset_flat_stacks, length_flat_stacks


@njit
def mutation_ops(V1, T_arr, rng=None):
  """Mutation based on assembly operation"""
  assembly_ops = get_assembly_op(T_arr) 
  dim_of_stack = 3         # op, machine, factory (tri-chain)
  # print(f"assembly_ops")
  # print(assembly_ops)

  # rng = np.random.default_rng()
  # selected_gene = rng.choice([int(ops[1:]) for ops in assembly_ops])
  idx_selected_gene = rng.integers(0, len(assembly_ops), dtype=np.int32)
  selected_gene = assembly_ops[idx_selected_gene]
  # print(f"selected_gene: ", selected_gene)

  # child_nodes = list(T.predecessors(f"W{selected_gene}"))
  child_nodes = np.array(
    [idx for idx, cond in enumerate(T_arr == selected_gene) if cond], dtype=np.int32)
  # print(f"child_nodes: ", child_nodes)

  # -- Create a lookup nodes for each stack
  # all_child_descendants = {int(node[1:]): [int(node[1:])] + [int(d[1:]) for d in nx.ancestors(T, node)] 
  #                         for node in child_nodes}
  all_child_descendants, offset_all_child_descendants, \
    length_all_child_descendants = create_stack_all_children(T_arr, child_nodes)
  # print(f"all_child_descendants = ",all_child_descendants)
  # print(f"offset_all_child_descendants = ",offset_all_child_descendants)
  # print(f"length_all_child_descendants = ",length_all_child_descendants)


  # all_stacks = {int(node[1:]): [] for node in child_nodes}
  all_stacks = np.empty((dim_of_stack, 0), dtype=np.int32)
  offset_all_stacks = np.zeros(len(child_nodes), dtype=np.int32)
  length_all_stacks = np.zeros(len(child_nodes), dtype=np.int32)

  # Create a stack for each child node
  idx_selected_gene = np.argwhere(V1[0, :] == selected_gene)[0,0]
  # print(f"segment= ", V1[0, 0:idx_selected_gene+1])
  for i in range(idx_selected_gene, -1, -1):
    op, machine, factory = V1[:, i]
    for idx_offset in range(len(offset_all_child_descendants)):
      offset_all_children = offset_all_child_descendants[idx_offset]
      length_all_children = length_all_child_descendants[idx_offset]
      stack_membership = all_child_descendants[
        offset_all_children:offset_all_children+length_all_children]
      if op in stack_membership:
        # split all_stacks at offset of op
        offset_stack = offset_all_stacks[idx_offset]
        length_stack = length_all_stacks[idx_offset]
        left_flat_stacks = all_stacks[:, 0:offset_stack+length_stack]
        right_flat_stacks = all_stacks[:, offset_stack+length_stack:]
        # print(f"op: ", op)
        # print(f"  offset_all_stacks", offset_all_stacks)
        # print(f"  length_all_stacks", length_all_stacks)
        # print(f"  all_child_descendants\n", all_child_descendants)
        # print(f"  left_flat_stacks\n", left_flat_stacks)
        # print(f"  right_flat_stacks\n", right_flat_stacks)
        temp = left_flat_stacks.copy()
        left_flat_stacks = np.zeros((dim_of_stack, len(temp[0])+1), dtype=np.int32)
        left_flat_stacks[:, :-1] = temp.copy()
        # left_flat_stacks[0, -1] = op
        # left_flat_stacks[1, -1] = factory
        left_flat_stacks[:, -1] = [op, machine, factory]
        all_stacks = np.append(left_flat_stacks, right_flat_stacks, axis=1)
        
        offset_all_stacks[idx_offset+1:] += 1
        length_all_stacks[idx_offset] += 1
        # print(f"  --> after")
        # print(f"  offset_all_stacks", offset_all_stacks)
        # print(f"  length_all_stacks", length_all_stacks)
        # print(f"  all_child_descendants\n", all_child_descendants)
        # print(f"  left_flat_stacks\n", left_flat_stacks)
        # print(f"  right_flat_stacks\n", right_flat_stacks)

  # print(f"all_stacks: \n", all_stacks)   # the first element in each child node is a top element in each stack
  # print(f"offset_all_stacks: ", offset_all_stacks)
  # print(f"length_all_stacks: ", length_all_stacks)

  # Pick randomly top element in each stack until all_stacks is empty
  # rng = np.random.default_rng()

  # mutated_stack = np.empty((dim_of_stack, 0), dtype=np.int32)
  mutated_stack = np.zeros((dim_of_stack, 1), dtype=np.int32)

  while True:
    idx_stack_key_selected = rng.integers(0, len(offset_all_stacks), dtype=np.int32)
    # print("  idx_stack_key_selected: ", idx_stack_key_selected)
    
    # length of the array in this stack is not zero
    if length_all_stacks[idx_stack_key_selected] != 0:
      offset_stack = offset_all_stacks[idx_stack_key_selected]
      length_stack = length_all_stacks[idx_stack_key_selected]
      # left_all_stacks = all_stacks[:, 0:offset_stack+length_stack]
      left_all_stacks = all_stacks[:, 0:offset_stack].copy()

      # right_all_stacks  = all_stacks[:, offset_stack+length_stack:]
      right_all_stacks  = all_stacks[:, offset_stack+1:].copy()
      # pop_op = left_all_stacks[:, 0].copy()
      pop_op = all_stacks[:, offset_stack].copy()
      mutated_stack = np.append(mutated_stack, pop_op.reshape((dim_of_stack, -1)), axis=1)
      all_stacks = np.append(left_all_stacks, right_all_stacks, axis=1)

      # print(f"  left_all_stacks: \n", left_all_stacks)
      # print(f"  right_all_stacks: \n", right_all_stacks)
      # print(f"  all_stacks: \n", all_stacks)
      # print(f"  pop_op: ", pop_op)

      length_all_stacks[idx_stack_key_selected] -= 1

      # if (idx_stack_key_selected < len(offset_all_stacks) - 1) \
      #     and (length_stack > 0):
      #   offset_all_stacks[idx_offset+1:] -= 1
      # -- all the stack after the selected stack will be shifted to the left
      offset_all_stacks[idx_stack_key_selected+1:] -= 1

      # print(f"    offset_all_stacks: ", offset_all_stacks)
      # print(f"    length_all_stacks: ", length_all_stacks)

    if sum(length_all_stacks) == 0:
      break
  
  # when we have zero length of stack in the stack_key_selected
  # remove the stack_key_selected from all_stacks
  # else:
    # length_all_stacks_2[idx_stack_key_selected] = 0
  mutated_stack = mutated_stack[:, 1:]
  # print(f"mutated_stack: \n", mutated_stack)


  # -- Create a Python array index for all descendants nodes of `selected_gene`
  descendants_selected_gene = np.sort(get_ancestors(T_arr, selected_gene))

  idx_descendants_selected_gene = np.empty(0, dtype=np.int32)
  for i in range(idx_selected_gene, -1, -1):
    op = V1[0, i]
    if op in descendants_selected_gene:
      idx_descendants_selected_gene = np.append(idx_descendants_selected_gene, [np.int32(i)])

  #print(idx_descendants_selected_gene)

  # -- Create mutated chromosome
  V1_prime = V1.copy()
  for mutate_gene, idx_mutate_gene in zip(mutated_stack.T, idx_descendants_selected_gene):
    V1_prime[:, idx_mutate_gene] = mutate_gene


  return V1_prime, len(mutated_stack)