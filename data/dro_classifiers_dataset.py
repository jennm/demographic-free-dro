import gc
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DROClassifiersDataset(Dataset):
    def __init__(self, dataset, process_item_fn, classifiers, model, n_classes,
                 group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.classifiers = classifiers
        self.tail_cache = dict()
        self.model = model
        self.get_hooks()
        self.n_groups = len(classifiers)
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        y_array = self.get_label_array()
        self._y_array = torch.LongTensor(y_array)

        self._group_array, self._group_counts = self.create_group_array(3)
        # self._group_array = torch.LongTensor(group_array)
        # self._group_counts = ((torch.arange(
            # self.n_groups).unsqueeze(1) == self._group_array).sum(1).float())

        self._y_counts = (torch.arange(
            self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def hook_fn(self, module, input, output):
        device = output.get_device()
        if device in self.tail_cache:
            self.tail_cache[device].append(input[0].clone().detach())
        else:
            self.tail_cache[device] = [input[0].clone().detach()]
    

    def get_hooks(self, num_embeddings=5):
        hooks = []
        num_layers = sum(1 for _ in self.model.modules())
        print('model num layers', num_layers)
        for i, module in enumerate(self.model.modules()):
            if i >= num_layers - num_embeddings:
                print(f"{i}: {num_layers - i}")
                hook = module.register_forward_hook(self.hook_fn)
                hooks.append(hook)

        print('hooks_length: ', len(hooks))

    def create_group_array(self, layer_num):
        groups = list()
        group_dict = dict()
        device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(0, self.n_groups + 1):
            group_dict[i] = 0
        for i in range(self._y_array.shape[0]):
            groups.append(list())
            unassigned_group = True

            # get embeddings
            inputs = self.dataset[i][0].unsqueeze(0)
            #print(inputs.shape)
            inputs = inputs.to(device)
            self.model(inputs)
            # embeddings = torch.cat(tuple([self.tail_cache[i][layer_num + 1].to(device) for i in list(self.tail_cache.keys())]), dim=0)
            # embeddings = embeddings.view(embeddings.size(0), -1)
            # self.tail_cache = dict()
            
            for j in range(len(self.classifiers)):
                # print(self.dataset[i])
                # will need to change later
                embeddings = torch.cat(tuple([self.tail_cache[i][j + 1].to(device) for i in list(self.tail_cache.keys())]), dim=0)
                embeddings = embeddings.view(embeddings.size(0), -1)
            
                embeddings = embeddings.to(device)
                classifier = self.classifiers[j]
                classifier = classifier.to(device)
                outputs = classifier(embeddings)
                _, predicted = torch.max(outputs, 1)
                # print(f'predicted {i}, {j}: {predicted}')
                if predicted[0] == 1:
                    groups[-1].append(j + 1)
                    unassigned_group = False
                    group_dict[j + 1] += 1
            if unassigned_group:
                # assuming that we assign everyone to a group
                groups[-1].append(0)
                group_dict[0] += 1
            
            del embeddings
            self.tail_cache = dict()
            gc.collect()
            torch.cuda.empty_cache()

        group_counts_list = list()
        for i in range(0, self.n_groups + 1):
            group_counts_list.append(group_dict[i])
        print('group_counts_list:', group_counts_list)
        return torch.LongTensor(groups), torch.FloatTensor(group_counts_list)
                
    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def get_group_array(self):
        if self.process_item is None:
            return self._group_array
        else:
            raise NotImplementedError

    def get_label_array(self):
        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

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
    else:  
        # Training and reweighting
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
