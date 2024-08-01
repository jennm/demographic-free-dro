import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import math

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

    return eigenvalues, eigenvectors

# shit in theory, maybe not in practice
# looking at eigenvalues can help us check if we're *really* finding k-subspace
def solve_sdp(A, V):
    n, d = A.shape[0], A.shape[1]
    # assert n == 25
    # assert d == 84
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

def find_V(A, k, tol=1e-3, max_iter=200):
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
    # make sure unit eigenvector
    V = eigenvectors[:, -k:]
    P = V @ V.T
    return P # eigenvectors[:, -k:]


def intra_group_distance(embeddings, metric='euclidean'):
    pairwise_distances = pdist(embeddings, metric=metric)
    return np.mean(pairwise_distances)


def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

def get_fair_subspace(G, subclasses, k, use_groups = range(25), use_misclassified = False, misclassified = None):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    A = []
    if use_misclassified:
        A.append(compute_covariance(G[misclassified]))
        A.append(compute_covariance(G[~misclassified]))
    else:
        for i in use_groups:
            G_i = G[subclasses == i]
            A.append(compute_covariance(G_i))
    A = np.array(A)

    k = min(min(np.linalg.matrix_rank(A)), k)

    print('Finding k:', k)
    Y_opt, _ = find_V(A, k)
    subspace = recover_subspace(Y_opt, k)

    return subspace


def get_fair_subspace_MANUAL_2_GROUPS(majority_group, minority_group, G, subclasses):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    A = compute_covariance(G)

    eigenvalues, eigenvectors = get_real_eig(A)

    G_1 = G[subclasses == majority_group] # M x D
    G_2 = G[subclasses == minority_group] # m x D

    variances_1 = []
    variances_2 = []
    variances_diffs = []

    for i in range(len(eigenvectors[0])):
        eigenvector = eigenvectors[:, i]
        variances_1.append(np.var(np.dot(G_1, eigenvector)))
        variances_2.append(np.var(np.dot(G_2, eigenvector)))
        variances_diffs.append(abs(variances_1[-1] - variances_2[-1]))
    
    std_dev = np.std(variances_diffs)
    print('std before:', std_dev)

    tolerance = std_dev / 10
    keep_idx = []
    keep_diffs = []
    for i in range(len(variances_1)):
        if abs(variances_1[i] - variances_2[i]) < tolerance: 
            keep_idx.append(i)
            keep_diffs.append(abs(variances_1[i] - variances_2[i]))

    print('std after:', np.std(keep_diffs))

    D = len(eigenvectors[0])

    P = np.zeros((D, D), dtype='float')
    for idx in keep_idx:
        u_i = eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    # U = np.concatenate([eigenvectors[:, idx].reshape(D, -1) for idx in keep_idx], axis=1)
    # G.dot(P) N x D
    # G.dot(U) N x k

    return P

def get_fair_subspace_MANUAL_N_GROUPS(groups, G, subclasses):
    groups.sort()

    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    A = compute_covariance(G)

    eigenvalues, eigenvectors = get_real_eig(A)

    G_i = []
    for i in groups:
        G_i.append(G[subclasses == i])

    D = eigenvectors.shape[1]

    variances_i = [[] for _ in range(len(groups))]
    for i in range(len(groups)):
        for j in range(D):
            eigenvector = eigenvectors[:, j]
            variances_i[i].append(np.var(np.dot(G_i[i], eigenvector)))

    # each row is the ith group, each column is how well the jth eigenvector explains the ith group

    variances_i = np.array(variances_i)
    
    abs_diff_sum = np.zeros(D)
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            abs_diff = np.abs(variances_i[i] - variances_i[j])
            abs_diff_sum = np.maximum(abs_diff_sum, abs_diff) # += abs_diff

    average_abs_diff = abs_diff_sum # / math.comb(len(groups), 2)

    std_dev = np.std(average_abs_diff)
    print('std before:', std_dev)

    tolerance = std_dev / 10

    print(average_abs_diff)
    print(tolerance)
    
    below_tolerance_indices = np.where(average_abs_diff < tolerance)[0]
    print('directions retained:', len(below_tolerance_indices))
    print('std after:', np.std(average_abs_diff[below_tolerance_indices]))

    eigenvectors = eigenvectors[:, below_tolerance_indices]

    P = np.zeros((D, D), dtype='float')

    for idx in range(eigenvectors.shape[1]):
        u_i = eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    return P

def get_fair_subspace_MANUAL_2_GROUPS_COSINE(majority_group, minority_group, G, subclasses):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    G_1 = G[subclasses == majority_group] # M x D
    G_2 = G[subclasses == minority_group] # m x D

    A_1 = compute_covariance(G)
    eigenvalues_1, eigenvectors_1 = get_real_eig(A_1)

    A_2 = compute_covariance(G_2)
    eigenvalues_2, eigenvectors_2 = get_real_eig(A_2)

    similarity_matrix = cosine_similarity(eigenvectors_1.T, eigenvectors_2.T)
    similarities = np.diag(similarity_matrix)

    threshold = np.std(similarities) / 10

    above_threshold_indices = np.where(similarities >= threshold)[0]

    print('directions retained:', len(above_threshold_indices))

    D = len(eigenvectors_1[0])

    eigenvectors_1 = eigenvectors_1[:, above_threshold_indices]
    eigenvectors_2 = eigenvectors_2[:, above_threshold_indices]

    average_eigenvectors = (eigenvectors_1 + eigenvectors_2) / 2


    P = np.zeros((D, D), dtype='float')
    # for idx in range(eigenvectors_1.shape[1]):
    #     u_i = eigenvectors_1[:, idx].reshape((D, 1))
    #     P += (u_i @ u_i.T)

    # for idx in range(eigenvectors_2.shape[1]):
    #     u_i = eigenvectors_2[:, idx].reshape((D, 1))
    #     P += (u_i @ u_i.T)

    for idx in range(average_eigenvectors.shape[1]):
        u_i = average_eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    return P

def get_fair_subspace_MANUAL_N_GROUPS_COSINE(groups, G, subclasses):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    # noise_scale = 0.001
    # noise = noise_scale * np.random.randn(*G.shape)
    # G = G + noise
    D = G.shape[1]
    groups.sort()

    group_eig = []
    for i in groups:
        G_i = G[subclasses == i]
        A_i = compute_covariance(G_i)
        _, eigenvectors_i = get_real_eig(A_i)
        group_eig.append(eigenvectors_i)


    similarities = np.zeros(D)
    denom = 0
    for i in range(len(groups)):
        for j in range(len(groups)):
            if i >= j: continue
            denom += 1
            # similarities += np.diag(cosine_similarity(group_eig[i].T, group_eig[j].T))
            similarities = np.maximum(similarities, np.diag(cosine_similarity(group_eig[i].T, group_eig[j].T)))

    # similarities /= math.comb(len(groups), 2)

    threshold = 0.5 # np.std(similarities) / 10
    # print(similarities)
    # print(threshold)
    # exit()

    above_threshold_indices = np.where(similarities >= threshold)[0]

    print('directions retained:', len(above_threshold_indices))

    for i in range(len(groups)):
        group_eig[i] = group_eig[i][:, above_threshold_indices]

    average_eigenvectors = np.mean(np.array(group_eig), axis=0)
    P = np.zeros((D, D), dtype='float')

    for idx in range(average_eigenvectors.shape[1]):
        u_i = average_eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    return P


    
    






'''
PCA on each group k vectors per grouped, ranked by eigenvalue
cosine similarity between corresponding vectors across groups
drop ones with low cosine similarity
play with threshold
'''