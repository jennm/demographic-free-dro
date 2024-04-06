import torch

from functools import partial
from torch.utils.data import DataLoader

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


def collate_func(batch, feature_extractor):
    inputs = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([sample[1] for sample in batch])
    group = torch.stack([sample[2] for sample in batch])
    data_idx = torch.stack([sample[4] for sample in batch])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        inputs = inputs.to(device)
        embeddings = feature_extractor(inputs)
        # NOTE: inputs, embeddings will be saved on GPU not CPU

    
    data = {'embeddings': embeddings, 'idx': data_idx, 'inputs': inputs, 'group': group, 'LR_targets': labels, 'actual_targets': labels}

    return data


def create_dataloader(feature_extractor, dataset, shared_dl_args):
    collate_fn = partial(collate_func, feature_extractor=feature_extractor)
    return DataLoader(dataset, **shared_dl_args, collate_fn=collate_fn)


def get_nodes(model, layers):
    train_nodes, _ = get_graph_node_names(model)
    nodes = {}
    num_layers = sum(1 for _ in model.modules())
    for i, module in enumerate(model.named_modules()):
        if (num_layers - i - 1) in layers:
            nodes[module[0]] = (num_layers - i - 1)
            assert module[0] in train_nodes

    return nodes


def get_embeddings(loader_kwargs, model, layers):
    nodes = get_nodes(model, layers)
    # TODO: move model to device in case we use model path that was saved when on GPU
    feature_extractor = create_feature_extractor(model, return_nodes=nodes)
    return feature_extractor


