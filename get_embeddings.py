import torch

from functools import partial
from torch.utils.data import DataLoader

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

import numpy as np


def collate_func(batch, feature_extractor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = torch.stack([sample[0] for sample in batch]).to(device)

    labels = torch.tensor([sample[1] for sample in batch]).to(device)
    LR_y = torch.tensor([sample[-2] for sample in batch]).to(device)

    group = np.array([sample[2] for sample in batch])
    classifier_group = np.array([sample[-1] for sample in batch])

    data_idx = np.array([sample[4] for sample in batch])


    with torch.no_grad():
        embeddings = feature_extractor(inputs)
        # NOTE: inputs, embeddings will be saved on GPU not CPU

        # if torch.all(LR_y == -1):
        #     predicted = torch.argmax(embeddings[str(0)], dim=1)
        #     misclassified = (predicted != labels).long()
        #     LR_y = misclassified

            # misclassified = torch.zeros(labels.shape, device=device)
            # num_ones = int(0.5 * misclassified.size(0))
            # indices = torch.randperm(misclassified.size(0))
            # misclassified[indices[:num_ones]] = 1
            # LR_y = misclassified.long()


    data = {'embeddings': embeddings, 'idx': data_idx, 
            'inputs': inputs, 'group': group, 'classifier_group' : classifier_group, 
            'LR_targets': LR_y, 'actual_targets': labels}

    return data


def create_dataloader(feature_extractor, dataset, sampler, shared_dl_args):
    collate_fn = partial(collate_func, feature_extractor=feature_extractor)
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
    # TODO: move model to device in case we use model path that was saved when on GPU
    feature_extractor = create_feature_extractor(model, return_nodes=nodes)
    return feature_extractor


