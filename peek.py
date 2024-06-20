import os
import pandas as pd

import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tqdm import tqdm
import matplotlib.pyplot as plt

# displays color map for corresponding color codes
# cmap = cm.get_cmap('hsv')
# cmap_vals = np.arange(0, 1, step=1 / 5)
# colors = []
# for ix in range(5):
#     rgb = cmap(cmap_vals[ix])[:3]
#     rgb = [int(float(x)) for x in np.array(rgb) * 255]
#     colors.append(rgb)


# # Visualize colors
# fig, ax = plt.subplots(figsize=(10, 2))

# for i, color in enumerate(colors):
#     rect = plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255)
#     ax.add_patch(rect)

# ax.set_xlim(0, len(colors))
# ax.set_ylim(0, 1)
# ax.set_xticks(np.arange(len(colors)) + 0.5)
# ax.set_xticklabels([f'{i % 10}, {(i+1) % 10}' for i in range(0, len(colors) * 2, 2)])#, rotation=90)
# ax.set_yticks([])

# plt.title('Generated Colors')
# plt.savefig('colors.png')

#####################################################################################################################

# sample imgs
# df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/coloredMNIST_HARD/data/metadata.csv')
# grouped_df = df.groupby(list(df.columns[2:]))
# sampled = grouped_df.apply(lambda x: x.sample(5, random_state=42))

# num_groups = len(sampled) // 5

# fig, axes = plt.subplots(nrows=num_groups, ncols=5, figsize=(15,12))

# for i in range(num_groups):
#     for j in range(5):
#         f = sampled.iloc[i * 5 + j][1]
#         image_path = 'coloredMNIST_HARD/data/colored_mnist_imgs' + '/' + f
#         img = plt.imread(image_path)
#         axes[i][j].imshow(img)
#         axes[i][j].axis('off')


# plt.tight_layout()
# plt.savefig('cmnist_med_sample_imgs.png')

######################################################################################################################

# df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/coloredMNIST_HARD/data/metadata.csv')

# df['train'] = df['split'].apply(lambda x: True if x == 0 else False)
# grouped = df.groupby('train')

# fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15,12))

# g = 0
# for name, group in grouped:
#     subgroup = group.groupby(list(df.columns[2:7]))
#     sampled = subgroup.apply(lambda x: x.sample(5, random_state=42))
#     for i in range(5):
#         for j in range(5):
#             f = sampled.iloc[i * 5 + j][1]
#             image_path = 'coloredMNIST_HARD/data/colored_mnist_imgs' + '/' + f
#             img = plt.imread(image_path)

#             axes[i * 2 + g][j].imshow(img)
#             axes[i * 2 + g][j].axis('off')

#     g += 1
    
# plt.tight_layout()
# plt.savefig('cmnist_med_sample_imgs.png')

####################################################################################################

df = pd.read_csv('results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv')
df = df[df['wrong_1_times'] == 1]

# sampled_df = df.sample(n=25, random_state=1)
# fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15,12))

# for i in range(5):
#     for j in range(5):
#         f = sampled_df.iloc[i * 5 + j][2]
#         image_path = 'coloredMNIST_HARD/data/colored_mnist_imgs' + '/' + f
#         img = plt.imread(image_path)

#         axes[i][j].imshow(img)
#         axes[i][j].axis('off')

# plt.tight_layout()
# plt.savefig('cmnist_hard_wrong_sample.png')

print('# Misclassified Points in Training Set:', len(df))
misclassified_groups = df.groupby(list(df.columns[3:13]))
print(df.columns[3:13].tolist())

minor_count = 0
minor_members = []
for _, group in misclassified_groups:
    group_size = len(group)
    example_row = group.iloc[0][3:13].tolist()
    print(f'Group: {example_row}; Count: {group_size}')

    a = [index for index, value in enumerate(example_row) if value == 1]
    if a[1] != a[0] + 5: 
        minor_count += 1
        minor_members.append(group_size)

sampled = misclassified_groups.apply(lambda x: x.sample(5, random_state=42))
num_groups = len(sampled) // 5

print('# Minority Groups Among Misclassified Points / # Total Groups:', minor_count, '/', num_groups)
print('Minority Group Counts Among Misclassified Points:', minor_members)
print('Total Minority Group Points Among Misclassified Points:', sum(minor_members))
print('Ratio of Minority Group Points Among Misclassified Points:', sum(minor_members) / len(df))

# df['spurious'] = (df['target'] == df['confounder'])
# print(np.sum(df['spurious']))
print('====================================================================================')
merged_csv = pd.read_csv('results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv')
merged_csv['spurious'] = (merged_csv['target'] == 1) & (merged_csv['confounder'] == 1)
print('# Points in Minority Group of Interest:', np.sum(merged_csv['spurious']))
merged_csv["our_spurious"] = merged_csv["spurious"] & merged_csv["wrong_1_times"]
print('# Misclassified Points Among Minority Group of Interest:', np.sum(merged_csv['our_spurious']))

spur_precision = np.sum(
            (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
        ) / np.sum((merged_csv[f"wrong_1_times"] == 1))
print("Spurious precision", spur_precision)
spur_recall = np.sum(
        (merged_csv[f"wrong_1_times"] == 1) & (merged_csv["spurious"] == 1)
    ) / np.sum((merged_csv["spurious"] == 1))
print("Spurious recall", spur_recall)

# fig, axes = plt.subplots(nrows=num_groups, ncols=5, figsize=(15,12))

# for i in range(num_groups):
#     for j in range(5):
#         f = sampled.iloc[i * 5 + j][2]
#         image_path = 'coloredMNIST_HARD/data/colored_mnist_imgs' + '/' + f
#         img = plt.imread(image_path)

#         axes[i][j].imshow(img)
#         axes[i][j].axis('off')

# plt.tight_layout()
# plt.savefig('cmnist_hard_all_wrong_groups_sample.png')

print('done')