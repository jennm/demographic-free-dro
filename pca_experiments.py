import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

from fair_k_pca import get_fair_subspace, get_fair_subspace_MANUAL_2_GROUPS, get_fair_subspace_MANUAL_N_GROUPS, get_fair_subspace_MANUAL_2_GROUPS_COSINE

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
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    if np.all(np.abs(np.imag(eigenvalues)) < 1e-10):
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    else:
        print('fucking hell')
        raise ValueError

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
    _, embeddings, subclasses, _, _, _ = unpack_data('train_14_epoch')
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    majority_subclass = 3
    minority_subclass = 1



    sample_size = 100

    G_1 = embeddings[subclasses == majority_subclass][:sample_size]
    G_2 = embeddings[subclasses == minority_subclass][:sample_size]
    # G_3 = embeddings[subclasses == 2][:sample_size]

    G = np.concatenate((G_1, G_2))

    u = get_fair_subspace(G, np.array([majority_subclass] * sample_size + [minority_subclass] * sample_size), 10, [minority_subclass, majority_subclass])
    # u = get_fair_subspace_MANUAL_2_GROUPS(6, 1, G, np.array([6] * sample_size + [1] * sample_size))
    # A = compute_covariance(G)
    # _, e = get_real_eig(A)
    # u = e[:, -2:]

    data = {
        'color': ['yellow 1s'] * 100 + ['blue 1s'] * 100
    }
    df = pd.DataFrame(data)

    projected_data = G.dot(u)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(projected_data)

    A = compute_covariance(projected_data)
    _, e = get_real_eig(A)
    u = e[:, -2:]

    projected_data = projected_data.dot(u)

    pca_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('PCA of Spurious Embeddings')
    plt.savefig('PCA_experiment.png')



    # # P = get_fair_subspace(embeddings, subclasses, 60, [3,8])
    # # P = get_fair_subspace_MANUAL_N_GROUPS(list(range(10)), embeddings, subclasses)
    # P = get_fair_subspace_MANUAL_2_GROUPS_COSINE(majority_subclass, minority_subclass, embeddings, subclasses)

    # print('Before - Intra G1:', intra_group_distance(G_1), 'Intra G2:', intra_group_distance(G_2), 'Inter:', inter_group_distance(G_1, G_2))
    # print('After - Intra G1:', intra_group_distance(G_1.dot(P)), 'Intra G2:', intra_group_distance(G_2.dot(P)), 'Inter:', inter_group_distance(G_1.dot(P), G_2.dot(P)))

    # print('Before - Intra G1:', intra_group_distance(G_1), 'Intra G3:', intra_group_distance(G_3), 'Inter:', inter_group_distance(G_1, G_3))
    # print('After - Intra G1:', intra_group_distance(G_1.dot(P)), 'Intra G3:', intra_group_distance(G_3.dot(P)), 'Inter:', inter_group_distance(G_1.dot(P), G_3.dot(P)))


# the misclassified points furthest in embedding space correspond to a spurious group

def experiment3():
    _, embeddings, subclasses, _, _, _ = unpack_data('train_14_epoch')

    g = [6]
    G = embeddings[np.isin(subclasses, g)]

    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    k = 1

    A = compute_covariance(G)
    v, e = get_real_eig(A)
    print(f'PCA{k} Top Eigenvalues:', v)
    pca = e[:, -k:].reshape(-1, k)

    u_space, pca_fair = get_fair_subspace(G, subclasses[np.isin(subclasses, g)], k, g)

    obj_fpca = np.trace((pca_fair @ pca_fair.T) @ A.T) 
    obj_pca1 = np.trace((pca @ pca.T) @ A.T)

    print('FPCA Objective Val:', obj_fpca)
    print(f'PCA{k} Objective Val:', obj_pca1)

    # print(np.allclose(pca1, pca1_fair, atol=1e-10))
    print('Cosnine Similarity of Last Vector', cosine_similarity(pca[:, -1].reshape(-1, 1).T, pca_fair[:, -1].reshape(-1, 1).T))

    if k > 1:
        intersection = pca.T @ pca_fair
        A_int = compute_covariance(intersection)
        v_int, e_int = get_real_eig(A_int)
        print('Top Eigenvalues', v_int[:-10])

    print('===========================================================================================')

experiment3()


'''
sub1 = spanX, sub2 = spanY
similarity(sub1, sub2) = eigenvalues of (X.T @ Y)
'''



