import numpy as np

from get_embeddings import create_dataloader
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def train_clusters(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    return kmeans


def unpack_data():
    train_npz = np.load('cmnist_meta_train.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    
    return predictions, embeddings, subclasses, labels, data_idx

def george_find_groups():
    # load data
    predictions, embeddings, subclasses, labels, data_idx = unpack_data()
    misclassified = np.argmax(predictions, axis=1) != labels

    # print(embeddings.shape)
    # print(misclassified.reshape(-1, 1).shape)
    # embeddings = np.hstack((embeddings, misclassified.reshape(-1, 1)))

    # subsample_indices = subsample(predictions, embeddings, subclasses, labels, data_idx)
    kmeans = train_clusters(embeddings, k=25)
    labels = kmeans.labels_

    print(set(labels))
    print(np.bincount(labels))

    print(labels.shape)

    # Create a DataFrame
    df = pd.DataFrame({'label': labels, 'group': subclasses})

    # Group by 'label' and 'class', and count the occurrences
    breakdown = df.groupby(['label', 'group']).size().unstack(fill_value=0)

    print(breakdown)

    # Iterate through each row and print the 5 largest values and their columns
    for index, row in breakdown.iterrows():
        largest_values = row.nlargest(5)
        print(f'Cluster {index + 1}:')
        for column, value in largest_values.iteritems():
            print(f'  {column}: {value}')
        print()

    # # Calculate the sum of each column and print it
    # column_sums = breakdown.sum()
    # print('Column Sums:')
    # print(column_sums)

    np.savez('classifier_groups.npz', group_array=labels)

    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(embeddings)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    # plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:,1], s=300, c='red', label='Centers')
    # plt.savefig('george.png')
    

















def __find_groups(train_data, val_data, aug_indices, feature_extractor, use_classifier_groups=False, num_epochs=5, k=0, max_iter=4, min_group=100, groups=None, **loader_kwargs):
    train_loader = create_dataloader(feature_extractor, train_data, None, loader_kwargs)
    val_loader = create_dataloader(feature_extractor, val_data, None, loader_kwargs)

    experiment(train_loader, 'train')
    experiment(val_loader, 'val')

def experiment(data_loader, desc):
    store_pred = []
    store_emb = []
    store_sub = []
    store_label = []
    store_data_idx = []
    store_loss = []

    for batch in data_loader:
        predictions = batch['embeddings'][str(0)]
        store_pred.append(predictions)

        embedding = batch['embeddings'][str(1)]
        store_emb.append(embedding)

        true_subclass = batch['group']
        store_sub.append(true_subclass)

        true_label = batch['labels']
        store_label.append(true_label)

        loss = batch['loss']
        store_loss.append(loss)

        data_idx = batch['idx']
        store_data_idx.append(data_idx)
    
    store_pred = np.concatenate(store_pred)
    store_emb = np.concatenate(store_emb)
    store_sub = np.concatenate(store_sub)
    store_label = np.concatenate(store_label)
    store_loss = np.concatenate(store_loss)
    store_data_idx = np.concatenate(store_data_idx)

    print((np.argmax(store_pred, axis=1) == store_label).sum() / len(store_label))

    np.savez(f'cmnist_meta_{desc}.npz', predictions=store_pred, embeddings=store_emb, subclass=store_sub, label=store_label, loss=store_loss, data_idx=store_data_idx)
