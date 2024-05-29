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

    group = np.array([sample[2] for sample in batch])
    data_idx = np.array([sample[4] for sample in batch])


    with torch.no_grad():
        gpu_inputs = torch.tensor(inputs).to(device)
        gpu_labels = torch.tensor(labels).to(device)
        embeddings = feature_extractor(gpu_inputs)
        loss = criterion(embeddings[str(0)], gpu_labels).detach().cpu().numpy()
        embeddings = {key: value.cpu().numpy() for key, value in embeddings.items()}
    
    data = {'embeddings': embeddings, 
            'idx': data_idx, 
            'inputs': inputs, 
            'group': group, 
            'labels': labels, 
            'loss': loss
            }

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
