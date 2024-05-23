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

# checks that all color classes are given only assigned majority or minority color code
# df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/coloredMNIST_HARD/data/metadata.csv')
# for i, r in df.iterrows():
#     target = np.where(r[2:7] == 1)[0][0]
#     color = np.where(r[7:] == 1)[0][0]
#     assert color == target or color == (target + 1) % 5
#     assert sum(r[2:]) == 2

#####################################################################################################################

# sample imgs
df = pd.read_csv('/scratch/09717/saloni/demographic-free-dro/coloredMNIST_HARD/data/metadata.csv')
grouped_df = df.groupby(list(df.columns[2:]))
sampled = grouped_df.apply(lambda x: x.sample(5, random_state=42))

num_groups = len(sampled) // 5

fig, axes = plt.subplots(nrows=num_groups, ncols=5, figsize=(15,12))

for i in range(num_groups):
    for j in range(5):
        f = sampled.iloc[i * 5 + j][1]
        image_path = 'coloredMNIST_HARD/data/colored_mnist_imgs' + '/' + f
        img = plt.imread(image_path)
        axes[i][j].imshow(img)
        axes[i][j].axis('off')


plt.tight_layout()
plt.savefig('cmnist_med_sample_imgs.png')

print('done')