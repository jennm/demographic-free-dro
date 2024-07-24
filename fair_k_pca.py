import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler

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

def solve_sdp(A, V):
    n, d = A.shape[0], A.shape[1]
    assert n == 25
    assert d == 84
    Y = cp.Variable((d, d), symmetric=True)
    objective = cp.Minimize(cp.trace(Y))
    constraints = [Y >=0, Y <= np.eye(d)]
    constraints += [cp.trace(Y @ A[i]) >= V for i in range(n)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status == cp.OPTIMAL:
        Y_opt = Y.value
        return Y_opt, np.trace(Y_opt)
    else:
        print(f'Naur: {problem.status}')
        return None, None

def find_V(A, k, tol=1e-3, max_iter=100):
    V_low = 0
    V_high = np.min([np.trace(A[i]) for i in range(len(A))]) - 0.001

    for _ in range(max_iter):
        V_mid = (V_low + V_high) / 2
        Y_opt, trace_Y_opt = solve_sdp(A, V_mid)

        if trace_Y_opt == None:
            V_high = V_high - 10
            continue

        print(trace_Y_opt, V_mid, 'V low', V_low, 'V high', V_high)

        if abs(trace_Y_opt - k) <= tol:
            print('found')
            return Y_opt, V_mid
        
        if trace_Y_opt > k:
            V_high = V_mid
        else:
            V_low = V_mid

    return None, None

def recover_subspace(Y, k):
    eigenvalues, eigenvectors = np.linalg.eigh(Y)
    return eigenvectors[:, -k:]


def intra_group_distance(embeddings, metric='euclidean'):
    pairwise_distances = pdist(embeddings, metric=metric)
    return np.mean(pairwise_distances)


def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

def get_fair_subspace(G, subclasses, k):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    d = G.shape[1]

    A = []
    for i in range(25):
        G_i = G[subclasses == i]
        A.append(compute_covariance(G_i))
    A = np.array(A)

    k = min(min(np.linalg.matrix_rank(A)), d)
    Y_opt, _ = find_V(A, k)
    subspace = recover_subspace(Y_opt, k)

    return subspace


def main():
    _, embeddings, subclasses, labels, _, _ = unpack_data('train')
    # target = [15, 16, 17, 18, 19]
    # confounder = [2, 7, 12, 17, 22]
    # union = target + confounder

    og_subclasses = np.copy(subclasses)

    # mask = np.isin(subclasses, union)
    # subclasses[~mask] = 0
    # subclasses[subclasses == 17] = 3
    # subclasses[np.isin(subclasses, target)] = 1
    # subclasses[np.isin(subclasses, confounder)] = 2

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    A = []
    for i in range(25):
        G_i = embeddings[subclasses == i]
        A.append(compute_covariance(G_i))
        # A.append(np.random.rand(84, 84) * 100)
    A = np.array(A)

    # rank = np.linalg.matrix_rank(A)
    # print(f"The rank of the matrix is: {rank}")
    # print(f"Rank of full A: {np.linalg.matrix_rank(compute_covariance(embeddings))}")

    k = 50
    Y_opt, _ = find_V(A, k)

    subspace = recover_subspace(Y_opt, k)
    print(subspace.shape)

    G_full = embeddings
    G_proj = G_full.dot(subspace)
    print(G_proj.shape)

    for i in range(15, 20):
        print(f'before intra {i}', intra_group_distance(G_full[og_subclasses == i][:100]))
        print('before intra 18', intra_group_distance(G_full[og_subclasses == i][:100]))
        print(f'after intra {i}', intra_group_distance(G_proj[og_subclasses == i][:100]))
        print('after intra 18', intra_group_distance(G_proj[og_subclasses == 18][:100]))
        print('before inter', inter_group_distance(G_full[og_subclasses == i][:100], G_full[og_subclasses == 18][:100]))
        print('after inter', inter_group_distance(G_proj[og_subclasses == i][:100], G_proj[og_subclasses == 18][:100]))


    data = {
        'subclass': ['17'] * 100 + ['18'] * 100,
    }
    df = pd.DataFrame(data)

    # scaler = StandardScaler()
    # G_proj = scaler.fit_transform(G_proj)
    A_proj = compute_covariance(G_proj)
    u_proj = recover_subspace(A_proj, 2)
    G_proj = G_proj.dot(u_proj) 

    pca_df = pd.DataFrame(data=np.concatenate((G_proj[og_subclasses == 17][:100][:, :2], G_proj[og_subclasses == 18][:100][:, :2]), axis=0), columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['subclass']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='subclass', data=pca_df, s=100)
    plt.title('PCA After Projecting Out Spurious Directions')
    plt.savefig('kFair_PCA_experiment.png')


    
# main()