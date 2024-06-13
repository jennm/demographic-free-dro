import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np

def load_data(algo_group_info_filename, meta_train_filename, meta_val_filename):
  data = torch.load(algo_group_info_filename)
  train = np.load(meta_train_filename)
  val = np.load(meta_val_filename)
  print(train.keys())
  groups = data['group_array']

  num_sub_groups = len(set(train['subclass'])) # all groups
  num_algo_groups = groups.shape[-1] # all groups including everything group from algo

  train_groups_count = [[0 for i in range(num_sub_groups)] for i in range(num_sub_groups)]

  train_total_groups = [0 for i in range(num_sub_groups)]
  groups = data['group_array']

  for i in range(len(train['data_idx'])):
    true_group = train['subclass'][i]
    for j in range(num_sub_groups):
      if groups[i][j] != -1:
        train_groups_count[true_group][j] += 1
    train_total_groups[true_group] += 1

  val_groups_count = [[0 for i in range(num_sub_groups)] for i in range(num_sub_groups + 1)]
  val_total_groups = [0 for i in range(num_sub_groups)]
  for i in range(len(val['data_idx'])):
    true_group = val['subclass'][i]
    for j in range(num_algo_groups):
      if groups[i + len(train['data_idx'])][j] != -1:
        val_groups_count[true_group][j] += 1
    val_total_groups[true_group] += 1
  return num_algo_groups, num_sub_groups, train_groups_count, train_total_groups, val_groups_count, val_total_groups

def create_graph(num_algo_groups, num_sub_groups, visualize, graph_filename, train_groups_count, train_total_groups, val_groups_count, val_total_groups):
  values = [list() for i in range(num_algo_groups)]

  if visualize == 'train':
    groups_count = train_groups_count
    groups_total = train_total_groups
  elif visualize == 'val':
    groups_count = val_groups_count
    groups_total = val_total_groups
  else:
    train_groups_count = np.array(train_groups_count)
    val_groups_count = np.array(val_groups_count)
    train_total_groups = np.array(train_total_groups)
    val_total_groups = np.array(val_total_groups)
    groups_count = train_groups_count + val_groups_count
    groups_total = train_total_groups + val_total_groups

  for i in range(num_algo_groups):
    results = f'{i}: '
    for j in range(num_sub_groups): # is this correct?
      results += f'{groups_count[j][i] / groups_total[j] * 100}\t'
    # print(results)



  for i in range(num_algo_groups):
    results = f'{i}: '

    for j in range(num_sub_groups):
      values[i].append(groups_count[j][i] / groups_total[j] * 100)
      results += f'{groups_count[j][i] / groups_total[j] * 100}\t'
    # print(values)



  categories = [i for i in range(num_sub_groups)]


  width = 0.5

  fig, ax = plt.subplots()
  color_arr= ['lightblue', 'orange', 'yellow', 'green']
  for i in range(num_algo_groups):
    ax.barh(categories, values[i], width, color=color_arr[i % 4], label=f"Group {i+1}")#, bottom=bottom)
    
    


  ax.set_xlabel('Percentage of group')
  ax.set_ylabel('Group identified by LR classifier')
  ax.set_title('Percentage of true groups identified by LR classifiers')
  ax.legend()
  plt.savefig(graph_filename)
  plt.show()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("-g", "--graph_filename", default="bar_graph.jpg")
  parser.add_argument("--algo_group_info", default='groups_from_classifiers_info_.pt')
  parser.add_argument("--meta_train_filename", default='cmnist_meta_train.npz')
  parser.add_argument("--meta_val_filename", default='cmnist_meta_val.npz')
  parser.add_argument("-v", "--visualize",choices=['train', 'val', 'all'], default="val")

  args = parser.parse_args()

  num_algo_groups, num_sub_groups, train_group_counts, train_total_groups, val_groups_count, val_total_groups  = load_data(args.algo_group_info, args.meta_train_filename, args.meta_val_filename)

  create_graph(num_algo_groups, num_sub_groups, args.visualize, args.graph_filename, train_group_counts, train_total_groups, val_groups_count, val_total_groups)