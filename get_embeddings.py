import torch

from functools import partial
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
from data.folds import Subset
from data import dro_dataset


from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler

from functools import partial
from collections import Counter

import numpy as np


def collate_func(batch, feature_extractor, criterion):  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    inputs = np.stack([sample[0] for sample in batch])
    labels = np.array([sample[1] for sample in batch])
    LR_y = np.array([sample[-2] for sample in batch])

    group = np.array([sample[2] for sample in batch])
    classifier_group = np.array([sample[-1] for sample in batch])

    data_idx = np.array([sample[4] for sample in batch])


    with torch.no_grad():
        gpu_inputs = torch.tensor(inputs).to(device)
        gpu_labels = torch.tensor(labels).to(device)
        embeddings = feature_extractor(gpu_inputs)
        loss = criterion(embeddings[str(0)], gpu_labels).detach().cpu().numpy()
        embeddings = {key: value.cpu().numpy() for key, value in embeddings.items()}
        # NOTE: inputs, embeddings will be saved on GPU not CPU    
    
    data = {'embeddings': embeddings, 'idx': data_idx, 
            'inputs': inputs, 'group': group, 'classifier_group' : classifier_group, 
            'LR_targets': LR_y, 'actual_targets': labels, 'loss': loss}

    return data


def create_dataloader(feature_extractor, dataset, sampler, shared_dl_args):
    collate_fn = partial(collate_func, feature_extractor=feature_extractor, criterion=nn.CrossEntropyLoss(reduction='none'))
    return DataLoader(dataset, **shared_dl_args, collate_fn=collate_fn, sampler=sampler)


def get_nodes(model, layers):
    train_nodes, _ = get_graph_node_names(model)
    nodes = {}

    num_layers = len(train_nodes)
    for layer in layers:
        if layer < num_layers:
            nodes[train_nodes[-(layer + 1)]] = layer

    return nodes


def get_embeddings(loader_kwargs, model, layers):
    nodes = get_nodes(model, layers)
    
    if 0 not in layers: layers.append(0)
    layers.sort()

    # TODO: move model to device in case we use model path that was saved when on GPU
    feature_extractor = create_feature_extractor(model, return_nodes=nodes)
    return feature_extractor


def get_subset(
    dataset,
    seed=0,
    fraction=0.2,
    use_classifier_groups=False
):
    random = np.random.RandomState(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    sz = int(math.ceil(len(indices) * fraction))
    indices = indices[:sz]
    split = Subset(dataset, indices)

    data = dro_dataset.DRODataset(
        split,
        process_item_fn=None,
        n_groups=dataset.n_groups,
        n_classes=dataset.n_classes,
        group_str_fn=partial(dataset.group_str, use_classifier_groups=use_classifier_groups),
        use_classifier_groups=use_classifier_groups
    )

    print('NEW SIZE', len(indices))

    return data


def get_upsampled_subset(
    dataset,
    misclassified_indices,
    seed=0,
    use_classifier_groups=False
):
    
    random = np.random.RandomState(seed)
    third = math.ceil(len(dataset) * 0.3)
    sub_misclassied_count = int(min(third * 0.5, len(misclassified_indices)))

    misclassified_indices = random.shuffle(misclassified_indices)
    sub_misclassified_indices = misclassified_indices[:sub_misclassied_count]

    indices = list(set(len(dataset)) - set(misclassified_indices))
    indices = random.shuffle(indices)

    indices = indices[:int(third * 0.5)]

    indices = indices + sub_misclassified_indices
    split = Subset(dataset, indices)

    data = dro_dataset.DRODataset(
        split,
        process_item_fn=None,
        n_groups=dataset.n_groups,
        n_classes=dataset.n_classes,
        group_str_fn=partial(dataset.group_str, use_classifier_groups=use_classifier_groups),
        use_classifier_groups=use_classifier_groups
    )

    print('NEW SIZE', len(indices))

    return data


# ignore_points corresponds to data_idx
def get_emb_loader(dataset, upweight_misclassified, feature_extractor, train=False, ignore_points=None, version='full + sampler', use_classifier_groups=False, **kwargs):
    sampler = None

    if upweight_misclassified is not None:
        dataset.update_LR_y(upweight_misclassified, np.ones(len(upweight_misclassified), dtype=np.int64))

    # Full train set, reweight sampler to overrepresent misclassified points
    if version == 'full + sampler':
        misclassified_count = len(upweight_misclassified)
        correct_count = len(dataset) - misclassified_count
        correct_wrong_weights = [len(dataset) / correct_count, len(dataset) / misclassified_count]
        weights = np.array([correct_wrong_weights[1] if i in upweight_misclassified else correct_wrong_weights[0] for i in range(len(dataset))])
        if ignore_points is not None: 
            ignore_points = np.where(np.isin(dataset.dataset.indices, ignore_points))[0]
            weights[ignore_points] = 0
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        subset = dataset

    # Subset identical distribution to train set, then apply sampler
    elif version == 'subset + sampler':
        subset = get_subset(dataset, seed=0, fraction=0.2, use_classifier_groups=False)
        subset_upweight_misclassified = list(set(subset.dataset.indices).intersection(upweight_misclassified))
        
        print('MISCLASSIFIED IN SUBSET', len(subset_upweight_misclassified))

        subset_misclassified_count = len(subset_upweight_misclassified)
        subset_correct_count = len(subset) - subset_misclassified_count
        subset_correct_wrong_weights = [len(subset) / subset_correct_count, len(subset) / subset_misclassified_count]

        subset_weights = [subset_correct_wrong_weights[1] if i in subset_upweight_misclassified else subset_correct_wrong_weights[0] for i in range(len(subset))]
        if ignore_points is not None: 
            ignore_points = np.where(np.isin(subset.dataset.indices, ignore_points))
            subset_weights[ignore_points] = 0
        sampler = WeightedRandomSampler(subset_weights, len(subset), replacement=True)

    # No sampler, subset skewed distribution to train set
    elif version == 'skewed subset':
        pass
        # subset = get_upsampled_subset(dataset, misclassified_indices=upweight_misclassified)
        # subset_upweight_misclassified = list(set(subset.dataset.indices).intersection(upweight_misclassified))
        
        # print('MISCLASSIFIED IN SUBSET', len(subset_upweight_misclassified))

        # subset_weights = [1] * range(len(subset))
        # if ignore_points is not None: 
        #     ignore_points = np.where(np.isin(subset.dataset.indices, ignore_points))
        #     subset_weights[ignore_points] = 0
        # sampler = WeightedRandomSampler(subset_weights, len(subset), replacement=True)

    if not train: sampler = None
    return subset, create_dataloader(feature_extractor, subset, sampler, kwargs), create_dataloader(feature_extractor, subset, None, kwargs), create_dataloader(feature_extractor, dataset, None, kwargs)