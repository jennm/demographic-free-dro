import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
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

def solve_sdp(A):
    n, d = A.shape[0], A.shape[1]
    X = cp.Variable((d, d), symmetric=True)
    objective = cp.Maximize(cp.trace(-np.eye(d) @ X))
    constraints = [cp.trace(-A[i] @ X) <= 1 for i in range(n)]
    constraints += [X >> 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status == cp.OPTIMAL:
        X_opt = X.value
        return X_opt
    else:
        print(f"Naur: {problem.status}")
        exit()

def extract_v(Y):
    eigenvalues, eigenvectors = np.linalg.eig(Y)
    index = np.argmax(eigenvalues)
    v = eigenvectors[:, index]
    v_normalized = v / np.linalg.norm(v)
    return v_normalized

def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

def main():
    _, embeddings, subclasses, labels, _, _ = unpack_data('train')
    A = []
    for i in range(25):
        G_i = embeddings[subclasses == i]
        A.append(compute_covariance(G_i))
    A = np.array(A)
    X_opt = solve_sdp(A)

    Y = X_opt / np.trace(X_opt)
    v = extract_v(Y)
    
    G_full = embeddings
    G_proj = G_full.dot(v)
    G_proj = np.vstack((G_proj, np.zeros_like(G_proj))).T

    print(inter_group_distance(G_full[subclasses == 0][:100], G_full[subclasses == 3][:100]))
    print(inter_group_distance(G_proj[subclasses == 0][:100], G_proj[subclasses == 3][:100]))
    
main()

'''
seems to work because the inter-group distance drops from ~8 to ~3.5
'''