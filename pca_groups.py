import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist, pdist, squareform

def unpack_data(part):
    train_npz = np.load(f'cmnist_meta_{part}.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    losses = train_npz['loss']
    
    return predictions, embeddings, subclasses, labels, data_idx, losses

def compute_covariance(G):
    G_bar = G.mean(axis=0, keepdims=True)
    G_centered = G - G_bar
    A = G_centered.T @ G_centered
    return A


def get_real_eig(A):
    assert np.all(np.isreal(A))
    assert np.allclose(A, A.T)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if np.all(np.abs(np.imag(eigenvalues)) < 1e-10):
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    return eigenvalues, eigenvectors

def intra_group_distance(embeddings, metric='euclidean'):
    pairwise_distances = pdist(embeddings, metric=metric)
    return np.mean(pairwise_distances)

def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

'''
Take a seed misclassified point
Look at its balanced neighborhood of misclassified points and correct
PCA on everyone
Look at variance between correctly classified and misclassified
Select features with high variance
Do u^T A to find the variances associated with each point (maybe double check with Kevin) [k x n where n is # of points]
Group the variances based on bins (or based on a certain threshold?) and assign them to a group. The groups in each kth component are disjoint but the groups across components can be overlapping
Run group dro (or your fav group inclusive algo) with the combination of all of these groups
'''
def experiment():
    # load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')
    misclassified = np.argmax(predictions, axis=1) != labels

    # G = embeddings
    # A = compute_covariance(G)
    # pca = PCA(n_components=1)
    # u = pca.fit_transform(A)

    # print(u.shape)

    # for i in range(25):
    #     G_i = G[subclasses == i]
    #     A_i = compute_covariance(G_i)
        
    print(labels[:10])
    print(subclasses[:10])

        

        # u.T @ A_i gives us a weighted sum of the covariances for each dimension
        # (1 x d) (d x d) => (1 x d) i.e. row vector of weighted sum of covariances for each dimension
        # then multiply the row vector by each weight in the principal component and sum to get another weighted sum
        # (1 x d) (d x 1) => (1 x 1) i.e



experiment()




# the misclassified points furthest in embedding space correspond to a spurious group