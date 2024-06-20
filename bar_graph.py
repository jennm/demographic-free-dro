import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np

data = torch.load('groups_from_classifiers_info_.pt')
train = np.load('cmnist_meta_train.npz')
val = np.load('cmnist_meta_val.npz')
groups = data['group_array']

num_sub_groups = groups.shape[-1] - 1
num_algo_groups = 10

train_groups_count = [[0 for i in range(num_algo_groups)] for i in range(num_sub_groups + 1)]
train_total_groups = [0 for i in range(num_sub_groups + 1)]
groups = data['group_array']
print(groups.shape)
print(groups.shape[-1])

for i in range(len(train['data_idx'])):
  true_group = train['subclass'][i]
  for j in range(num_sub_groups - 1):
    if groups[i][j + 1] != -1:
      train_groups_count[true_group][j] += 1
  train_total_groups[true_group] += 1

print('groups.shape: ', groups.shape)
val_groups_count = [[0 for i in range(num_sub_groups + 1)] for i in range(num_sub_groups + 1)]
val_total_groups = [0 for i in range(num_sub_groups + 1)]
for i in range(len(val['data_idx'])):
  true_group = val['subclass'][i]
  for j in range(num_sub_groups - 1):
    # print(i + len(train['data_idx']), j+1)
    if groups[i + len(train['data_idx'])][j + 1] != -1:
      val_groups_count[true_group][j] += 1
  val_total_groups[true_group] += 1

values = [list() for i in range(4)]
# values2 = list()
# values3 = list()
# values4 = list()
for i in range(5):
  results = f'{i}: '
  print(i)
  for j in range(4):
    # values[j].append(train_groups_count[j][i] / train_total_groups[j] * 100)
    results += f'{train_groups_count[j][i] / train_total_groups[j] * 100}\t'
  print(results)



print('val')
for i in range(num_sub_groups + 1):
  results = f'{i}: '

  print(i)
  for j in range(num_sub_groups):
    values[j].append(val_groups_count[j][i] / val_total_groups[j] * 100)
    results += f'{val_groups_count[j][i] / val_total_groups[j] * 100}\t'
  print(values)



categories = [i for i in range(1,num_sub_groups)]
values1 = values[0]
values2 = values[1]
values3 = values[2]
values4 = values[3]



width = 0.5

fig, ax = plt.subplots()
color_arr= ['lightblue', 'orange', 'yellow', 'green']
bottom = np.zeros(num_sub_groups + 1)
print(values)
print(values[0])
print(bottom)
for i in range(num_sub_groups):
  bottom += np.array(values[i])
  print(bottom)
  ax.barh(categories, bottom[0:2], width, color=color_arr[i % 4], label=f"Group {i+1}")
  
  


ax.set_xlabel('Percentage of group')
ax.set_ylabel('Group identified by LR classifier')
ax.set_title('Percentage of true groups identified by LR classifiers')
ax.legend()
plt.savefig("bar_graph.jpg")
plt.show()