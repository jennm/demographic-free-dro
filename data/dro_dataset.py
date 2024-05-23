import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from functools import partial
from collections import Counter

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes,
                 group_str_fn, use_classifier_groups=False, new_up_weight_array=None):

        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_classes = n_classes
        self.group_str = partial(group_str_fn, use_classifier_groups=use_classifier_groups)

        group_array = []
        y_array = []
        
        self.n_groups = n_groups
        group_array = self.get_group_array(use_classifier_groups=use_classifier_groups)
        y_array = self.get_label_array()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._group_array = torch.tensor(group_array, dtype=torch.long, device=device)
        self._y_array = torch.LongTensor(y_array)
        
        if new_up_weight_array is None:
            try: 
                self.up_weight_array = self.dataset.get_up_weight_array()
            except:
                self.up_weight_array = torch.ones(len(self._y_array))
        else:
            self.update_up_weight_array(new_up_weight_array)
        

        keys, counts = torch.unique(self._group_array, sorted=True, return_counts=True)
        counts = {k : v for k, v in zip(keys.cpu().numpy(), counts.cpu().numpy())}
        self._group_counts = torch.tensor([counts.get(i, 0) for i in range(self.n_groups)], dtype=torch.long, device=device)
        # self._group_counts = torch.unique(self._group_array, sorted=True, return_counts=True)[1]
        # if len(self._group_counts) != self.n_groups:
        #     self._group_counts = self._group_counts[1:]

        self._y_counts = (torch.arange(
            self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def update_up_weight_array(self, new_up_weight_array):
        self.dataset.update_up_weight_array(new_up_weight_array)

    def get_group_array(self, use_classifier_groups):
        if self.process_item is None:
            return self.dataset.get_group_array(use_classifier_groups=use_classifier_groups) 
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

    def group_counts(self): # TODO: check this
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, y, g, _ in self:
            return x.size()

    def create_LR_y(self):
        self.dataset.create_LR_y()

    def get_LR_label_array(self):
        return self.dataset.get_LR_label_array()

    def update_LR_y(self, idx, new_y):
        self.dataset.update_LR_y(idx, new_y)

def get_loader(dataset, train, reweight_groups, upweight_misclassified, **kwargs):
    if not train:  # Validation or testing
        assert reweight_groups is None
        shuffle = False
        sampler = None
    elif upweight_misclassified is not None:
        print('UPWEIGHT MISCLASSIFIED')
        misclassified_count = len(upweight_misclassified)
        correct_count = len(dataset) - misclassified_count
        correct_wrong_weights = [len(dataset) / correct_count, len(dataset) / misclassified_count]
        weights = [correct_wrong_weights[1] if i in upweight_misclassified else correct_wrong_weights[0] for i in range(len(dataset))]
        shuffle = False
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
    elif not reweight_groups:  # Training but not reweighting
        shuffle = True
        sampler = None
    else:  # Training and reweighting
        # When the --robust flag is not set, reweighting changes the loss function
        # from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
        # When the --robust flag is set, reweighting does not change the loss function
        # since the minibatch is only used for mean gradient estimation for each group separately
        group_weights = torch.where(dataset._group_counts == 0, torch.tensor(0), len(dataset) / dataset._group_counts)
        if len(dataset._group_array.shape) > 1:
            intersection_counts = (dataset._group_array.unsqueeze(0) == dataset._group_array.unsqueeze(1)).all(dim=2).sum(dim=1)
            weights = len(dataset) / intersection_counts

        else:
            weights = group_weights[dataset._group_array] # _group_array comes from dataset.get_group_array(use_classifier_groups)
        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights,
                                        len(dataset),
                                        replacement=True)
        shuffle = False

    # assert shuffle == False
    loader = DataLoader(dataset, shuffle=shuffle, sampler=sampler, **kwargs) # NOTE: __getitem__ will just return both group ids
    return loader
