# Logs
# - [2024/11/10]   
#   This is a copy of `geneticAlgo/utils.py` and some parts
#   of `utils_twoFacs.py`

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from numba import njit
from matplotlib.patches import ConnectionPatch

@njit
def array_copy(arr):
  return arr[:]

def read_process_machine_time(filename):
  df_process_machine_time = pd.read_csv(filename)
  return df_process_machine_time 

def read_process_priority(filename):
  df_process_priority = pd.read_csv(filename)
  return df_process_priority

def read_transfer_time(filename):
  df = pd.read_csv(filename)
  num_of_factory = int(filename.split(".")[-2].split("-")[-1])
  # display(df)
  # print(num_of_factory)

  fac_fac_time_matrix = np.zeros((num_of_factory, num_of_factory), dtype=np.int32)
  for row in df.values:
    # print(row)
    fac_fac_time_matrix[row[0]-1, row[1]-1] = row[2]
    fac_fac_time_matrix[row[1]-1, row[0]-1] = row[2]
  return fac_fac_time_matrix


def get_edges(df_process_priority):
  list_process_priority = df_process_priority.values.tolist()
  # print(f"list_process_priority")
  # print(list_process_priority)
  all_edges_arr = [[[int(p[1:])-1, int(c_i[1:])-1] for c_i in c.split(",")] 
      for p, c in list_process_priority]
  return np.concatenate(all_edges_arr, dtype=np.int32)


def get_tree_struct(all_edges):
  T = nx.DiGraph()

  for edges in all_edges[::-1]:
    T.add_edges_from([edges])

  T = T.reverse()
  # -- store T in the format of parrent-array representation
  #  node0 node1  node2  node3
  # [  -1,     0,     0,     0, ...]
  # node 1, 2, and 3 point to node0
  
  root = 0
  T_arr = np.full(len(T), -1, dtype=np.int32)
  for node in T.nodes():
    if node != root:
      T_arr[node] = list(T.successors(node))[0]

  return T_arr, T

def draw_tree_struct(T, pos=None, figsize=(6, 4), node_size=900, font_size=9):
  # T.graph["graph"] = {"dir": "back"}
  if pos == None:
    T.graph["graph"] = {"rankdir": "BT"}
    # pos = nx.nx_agraph.graphviz_layout(T, prog="dot")

    # automatic node pos with graphviz_layout
    pos = nx.drawing.nx_agraph.graphviz_layout(T, prog="dot",
      args="-Grankdir=BT")

  # manual node pos
  # pos = {
  #   "W9": (1, 0), "W5": (3, 0),
  #   "W4": (2, 1), "W8": (4, 1),
  #   "W3": (2, 2), "W7": (4, 2),
  #   "W10": (1, 3), "W2": (2, 3), "W6": (4, 3), "W11": (5, 3),
  #   "W1": (3, 4)
  # }

  fig, ax = plt.subplots(figsize=figsize)
  nx.draw(T, pos, ax=ax, node_size=node_size, with_labels=True, node_color="w", 
          edgecolors="k", font_size=font_size)
  plt.show(fig)

def get_p_ik(df_process_machine_time, num_of_machine):

  # num_of_machine = len(dict_process_machine_time["W1"][0])
  # print(num_of_machine)

  # -- processing time in each operation represent as a row that contains
  #    an array with the length is equal to num_of_machine
  #    For the non-zero element represents available machine for the given
  #    oepration
  p_ik = np.zeros((len(df_process_machine_time), num_of_machine), dtype=np.uint32)
  
  for op, *M in df_process_machine_time.values:
    # machine_row = np.zeros(num_of_machine, dtype=np.uint32)
    # machine_row[M-1] = p
    p_ik[int(op[1:])-1] = M

  return p_ik

def get_params_input(df_process_priority, df_process_machine_time, num_of_machine):
  # -- Processing time of operation O_i on machine k
  p_ik = get_p_ik(df_process_machine_time, num_of_machine)
  all_edges = get_edges(df_process_priority)
  T = nx.DiGraph()

  for edges in all_edges[::-1]:
    T.add_edges_from([edges])
  # T.graph["graph"] = {"dir": "back"}
  T = T.reverse()

  # -- Get all leaf nodes and non-leaf nodes
  leaf_nodes_arr = np.zeros(len(T), dtype=np.int8)
  for node in T.nodes: 
    # print(node, T.in_degree(node))
    if T.in_degree(node) == 0:
      leaf_nodes_arr[node] = 1
  non_leaf_nodes_arr = np.abs(leaf_nodes_arr - 1, dtype=np.int8) # negation


  # -- Calculate all the remaining times R_i
  root = [n for n, d in T.out_degree() if d==0][0]

  arr_R_i = np.zeros(len(T), dtype=np.uint32)
  for node in T.nodes():
    # print(non_root_node)
    path_to_root = list(nx.shortest_path(T, source=node, target=root))
    all_bar_p_i = []
    for node_along in path_to_root:
      # idx_machine_of_op = np.argwhere(p_ik[node_along] != 0)[0][0]  # zeroth based
      proc_time = p_ik[node_along].copy()
      max_proc_time = np.max(proc_time)

      # set the non-available machine to have max_proc_time + 1
      # (greater than max_proc_time)
      proc_time[proc_time == 0] = max_proc_time + 1

      # we use minimum proc_time to find out what is the best
      # scenario for makespan by selecting only the smallest
      # proc_time in each node along path_to_root
      idx_min_time = np.argmin(proc_time)
      processing_time = p_ik[node_along][idx_min_time]
      all_bar_p_i.append(processing_time)

    R_i = sum(all_bar_p_i)
    # R_i
    arr_R_i[node] = R_i


  # -- Number of available machines for operation O_i
  # In the paper they use this notation using M_i.
  idx_Pi_i = np.zeros((len(T), num_of_machine), dtype=np.uint8)
  for op, *M in df_process_machine_time.values:
    M = np.array(M, dtype=np.int32)
    M[M != 0] = 1
    idx_Pi_i[int(op[1:])-1] = M


  params_input = {
    "Psi_1": leaf_nodes_arr.copy(),
    "Psi_2": non_leaf_nodes_arr.copy(),
    "Psi_0": np.zeros(len(T), dtype=np.int8),
    "R_i": arr_R_i.copy(),
    "idx_Pi_i": idx_Pi_i.copy(),   # In paper, they used M_i
    "p_ik": p_ik.copy(),
    "n": len(T.nodes)
  }

  return params_input


def plot_gantt_chart(starting_time_operation_in_machine, num_of_factory, 
                      num_of_machine, p_ik, fontsize=16, width=8):
      
  fig, axes = plt.subplots(nrows=num_of_factory, ncols=1, 
                            figsize=(width, 3*num_of_factory), sharex=True)

  makespan = 0
  for op, (starttime, machine, fac_key) in enumerate(starting_time_operation_in_machine.T):

    # k is machine, v is pair of operation and its starting time
    # print(op, starttime)
    # print(p_ik[op][k-1])
    if makespan <= starttime + p_ik[op, machine-1]:
      makespan = starttime + p_ik[op, machine-1]
    
    axes[fac_key-1].barh(machine, p_ik[op, machine-1], left=starttime, 
                          height=0.5, color="None", edgecolor="k")
    
    x_text_coor = starttime + p_ik[op, machine-1]/2.
    # axes[fac_key-1].text(x_text_coor, k, r"$O_{{{:}}}$".format(op), 
    #                       ha="center", va="center", fontsize=fontsize)
    axes[fac_key-1].text(x_text_coor, machine, r"${{{:}}}$".format(op), 
                          ha="center", va="center", fontsize=fontsize)


  # axes[fac_key-1].invert_yaxis()
  for fac_key in range(1, num_of_factory+1):
    machine_range = np.arange(num_of_machine, dtype=np.int8) + 1  
    axes[fac_key-1].set_yticks(machine_range)
    axes[fac_key-1].set_yticklabels([r"$\mu_{:}$".format(machine_num) 
                                      for machine_num in machine_range])

    axes[fac_key-1].set_ylabel("Machine")
    axes[fac_key-1].set_title(f"Factory {fac_key}")

    axes[fac_key-1].grid("on")
  
  # -- adding makespan
  y_low = axes[-1].get_ylim()[0]
  y_high = axes[0].get_ylim()[1]
  makespan_line = ConnectionPatch(
    xyA=[makespan, y_low], xyB=[makespan, y_high], 
    coordsA=axes[-1].transData, coordsB=axes[0].transData, 
    linestyle="--")

  # bound_last_ax = axes[-1].get_position().bounds
  # bound_second_to_last_ax = axes[-2].get_position().bounds

  # y_makespan = 0.5*(bound_last_ax[1] + bound_last_ax[3] + bound_second_to_last_ax[1])
  y_makespan_offset = 0.5
  y_makespan = axes[-1].get_ylim()[1]
  fig.add_artist(makespan_line)

  axes[-1].text(makespan, y_makespan + y_makespan_offset, r"{}".format(makespan),
                ha="center", va="center", fontsize=fontsize, backgroundcolor="w")

  axes[-1].set_xlabel(r"Time period, $T$ (hour)")
  plt.subplots_adjust(hspace=0.4)
  plt.show(fig)