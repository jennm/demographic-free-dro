import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes,
                 group_str_fn, new_up_weight_array=None):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        group_array = self.get_group_array()

        y_array = self.get_label_array()

        self._group_array = torch.LongTensor(group_array)

        self._y_array = torch.LongTensor(y_array)
        
        if new_up_weight_array is None:
            try: 
                self.up_weight_array = self.dataset.get_up_weight_array()
            except:
                self.up_weight_array = torch.ones(len(self._y_array))
        else:
            self.update_up_weight_array(new_up_weight_array)

        self._group_counts = torch.unique(self._group_array, sorted=True, return_counts=True)[1]
        self._group_counts = torch.cat((self._group_counts[1:], self._group_counts[0].unsqueeze(0)))

        self._y_counts = (torch.arange(
            self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def update_up_weight_array(self, new_up_weight_array):
        self.dataset.update_up_weight_array(new_up_weight_array)

    def get_group_array(self):
        if self.process_item is None:
            return self.dataset.get_group_array()
        else:
            raise NotImplementedError

    def get_label_array(self):
        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

    def get_up_weight_array(self):
        return self.up_weight_array

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g, _ in self:
            return x.size()


def get_loader(dataset, train, reweight_groups, **kwargs):
    if not train:  # Validation or testing
        assert reweight_groups is None
        shuffle = False
        sampler = None
    elif not reweight_groups:  # Training but not reweighting
        shuffle = True
        sampler = None
    else:  # Training and reweighting
        # When the --robust flag is not set, reweighting changes the loss function
        # from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
        # When the --robust flag is set, reweighting does not change the loss function
        # since the minibatch is only used for mean gradient estimation for each group separately
        group_weights = len(dataset) / dataset._group_counts
        weights = group_weights[dataset._group_array]
        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights,
                                        len(dataset),
                                        replacement=True)
        shuffle = False

    # assert shuffle == False
    loader = DataLoader(dataset, shuffle=shuffle, sampler=sampler, **kwargs)
    return loader
