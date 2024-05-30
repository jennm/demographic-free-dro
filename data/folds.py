import numpy as np
import torch

from data import dro_dataset

import bisect
import warnings

from torch._utils import _accumulate
from torch import randperm, default_generator

from functools import partial


class Subset(torch.utils.data.Dataset): # Subset goes directly ontop of the underlying dataset (like celebA, mnist, etc)
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices, use_classifier_groups=False):
        self.dataset = dataset
        self.indices = indices
        self.group_array = self.get_group_array(re_evaluate=True, use_classifier_groups=use_classifier_groups)
        self.label_array = self.get_label_array(re_evaluate=True)
        
    def update_up_weight_array(self, new_up_weight_array):
        self.up_weight_array = self.dataset.update_up_weight_array(new_up_weight_array)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True, use_classifier_groups=False): # TODO: make sure looks at right group_array
        """Return an array [g_x1, g_x2, ...]"""
        # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
        if re_evaluate:
            group_array = self.dataset.get_group_array(use_classifier_groups=use_classifier_groups)[self.indices] # if we're using classifier groups, underlying call to get group array will do it's thing
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array

    def update_y(self, idx, new_y):
        self.dataset.update_y(idx, new_y)

    def get_n_groups(self, use_classifier_groups):
        return self.dataset.get_n_groups(use_classifier_groups)


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concate datasets

    Extends the default torch class to support group and label arrays.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_group_array(self, use_classifier_groups): # TODO: Check this
        group_array = []
        for dataset in self.datasets:
            group = dataset.get_group_array(use_classifier_groups)
            if type(group) is not torch.tensor:
                group = torch.tensor(group)
            group_array.append(group)
        group_array = torch.concat(group_array)
        return group_array

    def get_label_array(self):
        label_array = []
        for dataset in self.datasets:
            label_array += list(np.squeeze(dataset.get_label_array()))
        return label_array

def get_fold(
    dataset,
    fold=None,
    cross_validation_ratio=0.2,
    num_valid_per_point=4,
    seed=0,
    shuffle=True,
    use_classifier_groups=False
):
    """Returns (train, valid) splits of the dataset.

    Args:
      dataset (DRODataset): the dataset to split into (train, valid) splits.
      cross_validation_ratio (float): valid set size is this times the size of
          the dataset.
      num_valid_per_point (int): number of times each point appears in a
          validation set.
      seed (int): under the same seed, the output of this is guaranteed to be
          the same.
      shuffle (bool): whether to shuffle the training-set for cross validation
          or not (used for debugging can be removed later.)

    Returns:
      folds (list[list[[(DRODataset, DRODataset)]]): the (train, valid) splits.
          In each outer list, the inner list valid sets span the entire train
          set.  Each inner list is length: num_valid_per_point * 1 /
          cross_validation_ratio.
    """

    if fold is not None:
        indices = fold.split("_")[1:]
        sweep_ind = int(indices[0])
        fold_ind = int(indices[1])
        assert sweep_ind is None or sweep_ind < num_valid_per_point
        assert fold_ind is None or fold_ind < int(1 / cross_validation_ratio)

    valid_size = int(np.ceil(len(dataset) * cross_validation_ratio))
    num_valid_sets = int(np.ceil(len(dataset) / valid_size))

    random = np.random.RandomState(seed)

    all_folds = []
    for sweep_counter in range(num_valid_per_point):
        folds = []
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        else:
            print("\n" * 10, "WARNING, NOT SHUFFLING", "\n" * 10)
        for i in range(num_valid_sets):
            train_indices = indices[:i * valid_size] + indices[(i + 1) *
                                                               valid_size:]
            print("len(train_indices)", len(train_indices))
            train_split = Subset(dataset, train_indices, use_classifier_groups) # needs to know what groups to refer to

            valid_indices = indices[i * valid_size:(i + 1) * valid_size]
            print("len(valid_indices)", len(valid_indices))
            valid_split = Subset(dataset, valid_indices, False) # NOTE: changed to always False
            if sweep_counter == 0 and i == 0:
                print("train_split", train_split, "valid_split", valid_split)
            folds.append((train_split, valid_split))
        all_folds.append(folds)

    if fold is not None:
        train_data_subset, val_data_subset = all_folds[sweep_ind][fold_ind]
        
        # Wrap in DRODataset Objects
        # TODO: Make sure looks at right group array
        train_data = dro_dataset.DRODataset( 
            train_data_subset,
            process_item_fn=None,
            n_groups=dataset.get_n_groups(use_classifier_groups),
            n_classes=dataset.n_classes,
            group_str_fn=partial(dataset.group_str, use_classifier_groups=use_classifier_groups),
            use_classifier_groups=use_classifier_groups
        )
        

        val_data = dro_dataset.DRODataset(
            val_data_subset,
            process_item_fn=None,
            n_groups=dataset.get_n_groups(use_classifier_groups=False),
            n_classes=dataset.n_classes,
            group_str_fn=partial(dataset.group_str, use_classifier_groups=False),
            use_classifier_groups=False
        ) # NOTE: changed so val does NOT use classifier groups

        return train_data, val_data
    else:
        # TODO: change this to return DRODatasets not just list.
        return all_folds
