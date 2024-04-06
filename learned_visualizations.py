from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap
import numpy as np

def visualize(embeddings, layer):
    writer = SummaryWriter('runs/vis1')
    train_embeddings = embeddings['train_loader']
    
    all_embeddings = []
    all_images = []
    all_groups = []
    for batch in train_embeddings:
        batch_embeddings = batch['embeddings'][str(layer)].cpu()
        all_embeddings.append(batch_embeddings)
        batch_images = batch['inputs']
        all_images.append(batch_images)
        all_groups.append(torch.tensor(batch['group']))

    all_embeddings = torch.cat(all_embeddings)
    all_images = torch.cat(all_images)
    all_groups = torch.cat(all_groups)

    writer.add_embedding(all_embeddings, label_img=all_images)
    writer.close()

    pca(all_embeddings, all_groups, layer)
    umap_(all_embeddings, all_groups, layer)
    tsne(all_embeddings, all_groups, layer)
    

def pca(all_embeddings, all_groups, layer):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    num_groups = len(torch.unique(all_groups))
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    plt.figure(figsize=(10, 8))
    for i in range(num_groups):
        mask = all_groups == i
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], s=10, color=colors[i], label=f'Group {i}')
    
    plt.legend(title='Groups', loc='best')
    plt.title('PCA Visualization')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('PCA.png')


def umap_(all_embeddings, all_groups, layer):
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    num_groups = len(torch.unique(all_groups))
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    plt.figure(figsize=(10, 8))
    for i in range(num_groups):
        mask = all_groups == i
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], s=10, color=colors[i], label=f'Group {i}')

    plt.legend(title='Groups', loc='best')
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Dim 1')
    plt.ylabel('UMAP Dim 2')
    plt.savefig('UMAP.png')


def tsne(all_embeddings, all_groups, layer):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    num_groups = len(torch.unique(all_groups))
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    plt.figure(figsize=(10, 8))
    for i in range(num_groups):
        mask = all_groups == i
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], s=10, color=colors[i], label=f'Group {i}')

    plt.legend(title='Groups', loc='best')
    plt.title('TSNE Visualization')
    plt.xlabel('TSNE Dim 1')
    plt.ylabel('TSNE Dim 2')
    plt.savefig('TSNE.png')


def visualize_LR():
    pass


def visualize_LR_tensorboard():
    pass