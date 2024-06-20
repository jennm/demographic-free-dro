import numpy as np
from get_embeddings import create_dataloader
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def unpack_data():
    train_npz = np.load('cmnist_meta_train.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    losses = train_npz['loss']
    
    return predictions, embeddings, subclasses, labels, data_idx, losses

def loss_binning():
    # load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data()
    misclassified = np.argmax(predictions, axis=1) != labels

    num_bins = 25
    bin_edges = np.percentile(losses, np.linspace(0, 100, num_bins))


    # bin_edges = np.linspace(start=np.min(losses), stop=np.max(losses), num=25)
    bin_indices = np.digitize(losses, bin_edges, right=True)

    print('Num bins:', set(bin_indices))

    df = pd.DataFrame({'loss': losses, 'label': labels, 'bin': bin_indices, 'subclass': subclasses, 'misclassified': misclassified, 'data_idx': data_idx})

    explore = df.groupby(['bin', 'subclass']).size().unstack(fill_value=0)
    print(explore)

    for index, row in explore.iterrows():
        largest_values = row.nlargest(5)
        print(f'Cluster {index}:')
        for column, value in largest_values.iteritems():
            print(f'  {column}: {value}')
        print()

    np.savez('classifier_groups.npz', group_array=bin_indices)
