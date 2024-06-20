'''
precision = TP / (TP + FP) i.e. what portion of predicted positive is TP?
if precision = 50% and recall = 99% 

5% of the dataset is subclass X
if we capture pretty much all of that, then TP maps to 5% of the dataset
then we can conclude that 50% = 5% / (5% + FP) which means we're predicting
10% of the dataset to belong to subclass X, and we capture all of subclass X but
only 5% of points that are NOT subclass X
'''

'''
if precision = 20% and recall = 99%
20% = 5% / (5% + 15%)
'''

import numpy as np


fps = [f'group_members_{g}.npz' for g in [3, 15, 16, 17, 19]]
groups = [[] for _ in range(54000)]

for i, f in enumerate(fps):
    npz = np.load(f)
    predicted_positive_data_idx = npz['group']
    # print(np.where(predicted_positive_data_idx == 1))

    for j in range(54000):
        groups[j].append((i + 1) if predicted_positive_data_idx[j] else -1)



groups = np.array(groups)
np.savez('classifier_groups.npz', group_array=groups)