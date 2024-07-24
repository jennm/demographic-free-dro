import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def experiment():
    _, embeddings, subclasses, _, _, _ = unpack_data('train')
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    majority_subclass = 1
    minority_subclass = 3
    sample_size = 100

    G_1 = embeddings[subclasses == majority_subclass][:sample_size]
    G_2 = embeddings[subclasses == minority_subclass][:sample_size]
    G = np.concatenate((G_1, G_2), axis=0)

    A = compute_covariance(G)
    A_1 = compute_covariance(G_1)
    A_2 = compute_covariance(G_2)

    n_components = 2
    pca = PCA(n_components=n_components)
    u = pca.fit_transform(A)

    print('G1 var exp, component 1', u[:, 0].T @ A_1 @ u[:, 0])
    print('G2 var exp, component 1', u[:, 0].T @ A_2 @ u[:, 0])
    print('G1 var exp, component 2', u[:, 1].T @ A_1 @ u[:, 1])
    print('G2 var exp, component 2', u[:, 1].T @ A_2 @ u[:, 1])


    data = {
        'color': ['red'] * 100 + ['blue'] * 100
    }
    df = pd.DataFrame(data)

    projected_data = G.dot(u)

    pca_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('PCA of Spurious Embeddings')
    plt.savefig('PCA_experiment.png')

def experiment2():
# load data
    _, embeddings, subclasses, _, _, _ = unpack_data('train')
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    majority_subclass = 1
    minority_subclass = 3

    sample_size = 100

    G_1 = embeddings[subclasses == majority_subclass][:sample_size]
    G_2 = embeddings[subclasses == minority_subclass][:sample_size]
    G = np.concatenate((G_1, G_2), axis=0)

    A = compute_covariance(G)

    eigenvalues, eigenvectors = get_real_eig(A)

    variances_1 = []
    variances_2 = []
    variances_diffs = []

    for i in range(len(eigenvectors[0])):
        eigenvector = eigenvectors[:, i]
        variances_1.append(np.var(np.dot(G_1, eigenvector)))
        variances_2.append(np.var(np.dot(G_2, eigenvector)))
        variances_diffs.append(abs(variances_1[-1] - variances_2[-1]))

    x = list(range(len(eigenvectors)))

    std_dev = np.std(variances_diffs)
    print('std before:', std_dev)

    plt.scatter(x, variances_1)
    plt.scatter(x, variances_2)
    plt.savefig('variance_comparison.png')

    tolerance = std_dev / 10

    keep_idx = []
    keep_diffs = []
    for i in range(len(variances_1)):
        if abs(variances_1[i] - variances_2[i]) < tolerance: 
            keep_idx.append(i)
            keep_diffs.append(abs(variances_1[i] - variances_2[i]))
    
    print('directions retained', len(keep_idx))
    print('std after:', np.std(keep_diffs))

    D = len(eigenvectors[0])

    P = np.zeros((D, D), dtype='float')
    for idx in keep_idx:
        u_i = eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    data = {
        'subclass': ['red 1s'] * 100 + ['blue 1s'] * 100,
    }
    df = pd.DataFrame(data)

    projected_data = G.dot(P)

    G_proj_bar = projected_data.mean(axis=1, keepdims=True)
    G_proj_centered = projected_data - G_proj_bar
    A_proj = G_proj_centered.T @ G_proj_centered

    pca = PCA(n_components=2)
    u_proj = pca.fit_transform(A_proj)

    viz_projected_data = G_proj_centered.dot(u_proj)

    pca_df = pd.DataFrame(data=viz_projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['subclass']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='subclass', data=pca_df, s=100)
    plt.title('PCA After Projecting Out Spurious Directions')
    plt.savefig('PCA_experiment2.png')

    print('Before - Intra G1:', intra_group_distance(G_1), 'Intra G2:', intra_group_distance(G_2), 'Inter:', inter_group_distance(G_1, G_2))
    print('After - Intra G1:', intra_group_distance(G_1.dot(P)), 'Intra G2:', intra_group_distance(G_2.dot(P)), 'Inter:', inter_group_distance(G_1.dot(P), G_2.dot(P)))

def experiment3():
    _, embeddings, subclasses, _, _, _ = unpack_data('train')
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    G = embeddings
    A = compute_covariance(G)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvectors = eigenvectors[:, ::-1]

    print(eigenvalues)

    # traces = []
    # d = len(eigenvectors)
    # for i in range(d):
    #     e = eigenvectors[:, i]
    #     traces.append(e.T @ A @ e)
    
    # traces = traces / sum(traces) * 100

    # print(sorted(traces, reverse=True))
    # print(sum(traces))

    # pca = PCA()
    # U = pca.fit_transform(A)
    # print([f'{num*100:.2f}' for num in pca.explained_variance_ratio_])

    # print(U.shape)
    # print(eigenvectors.shape)


experiment3()

# the misclassified points furthest in embedding space correspond to a spurious group