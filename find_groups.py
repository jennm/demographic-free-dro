import torch
from classifier import LogisticRegressionModel
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors._classification import KNeighborsClassifier

from get_embeddings import get_emb_loader



'''
NOTE: The index that the DataLoader uses when getting a datapoint is the index relative to a fixed, static underlying data store.
Shuffling just changes the order in which indices are accessed. Split just changes the order in which indices are split.
Subset takes in a subset of indices from the full dataset and indexes relative to the full dataset. It does not do any reindexing.
'''

def reshape_batch_embs(device, batch):
    layers = batch.keys() # batch is really a dictionary of layer embeddings, get keys for all layers stored
    batch_multi_emb = [] # store flattened embeddings from each layer
    for layer in layers: # for each layer
        if layer == str(0): continue
        layer_emb = torch.tensor(batch[layer], device=device) # get the layer embeddings
        batch_multi_emb.append(layer_emb.view(layer_emb.size(0), -1)) # append the flattened embeddings i.e. (32 x 10 x 10 => 32 x 100)
    return torch.cat(batch_multi_emb, dim=1) # concatenate the flattened embeddings i.e. (32 x 10 and 32 x 20 => 32 x 30)

def find_groups(train_data, val_data, aug_indices, feature_extractor, use_classifier_groups=False, num_epochs=5, k=0, max_iter=4, min_group=100, groups=None, **loader_kwargs):
    if not groups: groups = defaultdict(lambda: [0])
    else: 
        torch.load(groups)
        groups = {i: row.tolist() for i, row in enumerate(groups)}

    pos_count, neg_count = float('inf'), float('inf')

    data = {}

    train_subset, train_subset_emb_loader, uni_train_subset_emb_loader, uni_train_emb_loader = get_emb_loader(train_data, aug_indices, feature_extractor, train=True, ignore_points=None, version='subset + sampler', use_classifier_groups=use_classifier_groups, **loader_kwargs)
    val_subset, val_subset_emb_loader, uni_val_subset_emb_loader, uni_val_emb_loader = get_emb_loader(val_data, aug_indices, feature_extractor, train=False, ignore_points=None, version='subset + sampler', use_classifier_groups=use_classifier_groups, **loader_kwargs)


    experiment(uni_train_emb_loader, 'train')
    experiment(uni_val_emb_loader, 'val')


def experiment(data_loader, desc):
    store_last = []
    store_pen = []
    store_sub = []
    store_label = []
    store_data_idx = []
    store_loss = []

    for batch in data_loader:
        last_layer_emb = batch['embeddings'][str(0)]
        # print(last_layer_emb.shape)
        store_last.append(last_layer_emb)
        pen_layer_emb = batch['embeddings'][str(1)]
        store_pen.append(pen_layer_emb)
        true_subclass = batch['group']
        store_sub.append(true_subclass)
        true_label = batch['actual_targets']
        store_label.append(true_label)
        loss = batch['loss']
        store_loss.append(loss)

        data_idx = batch['idx']
        store_data_idx.append(data_idx)
    
    store_last = np.concatenate(store_last)
    store_pen = np.concatenate(store_pen)
    store_sub = np.concatenate(store_sub)
    store_label = np.concatenate(store_label)
    store_loss = np.concatenate(store_loss)
    store_data_idx = np.concatenate(store_data_idx)

    print((np.argmax(store_last, axis=1) == store_label).sum() / len(store_label))

    # reducer = umap.UMAP()
    # store_pen = reducer.fit_transform(store_pen)

    np.savez(f'cmnist_meta_{desc}.npz', last_layer=store_last, pen_layer=store_pen, subclass=store_sub, label=store_label, loss=store_loss, data_idx=store_data_idx)

    
