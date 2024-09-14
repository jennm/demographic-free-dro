import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.linalg import qr, svd

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
    A /= 1/G.shape[0]
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


def solve_sdp(A, V, k):
    n, d = A.shape[0], A.shape[1]
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


def find_V(A, k, tol=1e-4, max_iter=1000):
    V_low = 0
    V_high = np.min([np.trace(A[i]) for i in range(len(A))]) - 0.00000001

    last_valid_Y = None

    for _ in range(max_iter):
        V_mid = (V_low + V_high) / 2
        Y_opt, trace_Y_opt = solve_sdp(A, V_mid, k)

        if trace_Y_opt == None:
            print('Something is probably going wrong.')
            continue

        last_valid_Y = Y_opt
        
        if abs(trace_Y_opt - k) <= tol:
            print('found')
            return Y_opt, V_mid
        
        if trace_Y_opt > k:
            V_high = V_mid
        else:
            V_low = V_mid

    return last_valid_Y, None


def recover_subspace(Y, k):
    d = Y.shape[0]
    eigenvalues, eigenvectors = get_real_eig(Y)
    V = eigenvectors[:, -k:]
    P = np.eye(d) - V @ V.T
    return P, Y


def intra_group_distance(embeddings, metric='euclidean'):
    pairwise_distances = pdist(embeddings, metric=metric)
    return np.mean(pairwise_distances)

def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

def get_fair_subspace(G, subclasses, k, use_groups = range(25), use_misclassified = False, misclassified = None):
    A = []
    if use_misclassified:
        A.append(compute_covariance(G[misclassified == 0]))
        A.append(compute_covariance(G[misclassified == 1]))
    else:
        for i in use_groups:
            G_i = G[subclasses == i]
            A.append(compute_covariance(G_i))
            if np.linalg.matrix_rank(A[-1]) != A[-1].shape[0]:
                A[-1] += np.random.normal(0, 1e-5, A[-1].shape)
    A = np.array(A)

    print('NUM GROUPS:', len(A))

    k = min(min(np.linalg.matrix_rank(A)), k)

    print('Finding k:', k, 'Min rank:', min(np.linalg.matrix_rank(A)))

    Y_opt, _ = find_V(A, k)
    subspace, span = recover_subspace(Y_opt, k)

    return subspace


def get_fair_subspace_MANUAL(g1, g2, k, G, subclasses, misclassified):
    G1 = scale_G(G[(subclasses == 0) & (misclassified == 1)]) # blue 1s
    G2 = scale_G(G[(subclasses == 1) & (misclassified == 0)]) # red 1s
    G3 = scale_G(G[(subclasses == 2) & (misclassified == 0)]) # blue 0s
    G4 = scale_G(G[(subclasses == 3) & (misclassified == 1)]) # red 0s

    print('How different are blue 1s from blue 1s?', np.mean(cosine_similarity(G1, G1))) # reds are pretty similar to each other
    print('How different are red 1s from red 1s?', np.mean(cosine_similarity(G2, G2))) # blues are pretty similar to each other
    print('How different are blue 1s from red 1s?', np.mean(cosine_similarity(G1, G2))) # reds and blues are pretty different to each other
    print('How different are blue 1s from blue 0s?', np.mean(cosine_similarity(G1, G3)))
    print('How different are red 1s from red 0s?', np.mean(cosine_similarity(G2, G4)))
   
    k = 30
    print('K:', k)

    G = np.concatenate((G1, G2), axis=0)
    A = compute_covariance(G)
    v, e = get_real_eig(A)
    pca1 = e[:,-k:].T

    G = np.concatenate((G3, G4), axis=0)
    A = compute_covariance(G)
    v, e = get_real_eig(A)
    pca2 = e[:, -k:].T

    C = pca1.T
    P = C @ np.linalg.pinv(C.T @ C) @ C.T
    P_fair1 = np.eye(84) - P

    C = pca2.T
    P = C @ np.linalg.pinv(C.T @ C) @ C.T
    P_fair2 = np.eye(84) - P

    def get_column_space(P):
        v, e = np.linalg.eigh(P)
        e_idx = np.isclose(v, 1, atol=1e-10)
        return e[:, e_idx]
    
    u1 = get_column_space(P_fair1)
    u2 = get_column_space(P_fair2)

    for i in range(len(u1[0])):
        a = u1[:, i].reshape(-1, 1).T
        b = u2[:, i].reshape(-1, 1).T
        print(cosine_similarity(a, b))

    overlap = u1.T @ u2
    canonical_correlations = np.sort(svd(overlap, compute_uv=False))

    P_fair = np.zeros((84, 84))
    for i in range(len(canonical_correlations)):
        if canonical_correlations[i] > 0.99:
            u = u1[:, i].reshape(84, 1)
            P_fair += (u @ u.T)

    # P_fair = P_fair1 + P_fair2

    G1_fair = scale_G(G1 @ P_fair)
    G2_fair = scale_G(G2 @ P_fair)
    G3_fair = scale_G(G3 @ P_fair)
    G4_fair = scale_G(G4 @ P_fair)

    # G1_fair = scale_G(G1 @ P_fair1)
    # G2_fair = scale_G(G2 @ P_fair1)
    # G3_fair = scale_G(G3 @ P_fair2)
    # G4_fair = scale_G(G4 @ P_fair2)

    print('How different are blue 1s from blue 1s when we look at the fair components?', np.mean(cosine_similarity(G1_fair, G1_fair))) # reds are pretty similar to each other
    print('How different are red 1s from red 1s when we look at the fair components?', np.mean(cosine_similarity(G2_fair, G2_fair))) # blues are pretty similar to each other
    print('How different are blue 1s from red 1s when we look at the fair components?', np.mean(cosine_similarity(G1_fair, G2_fair))) # reds and blues are pretty different to each other
    print('How different are blue 1s from blue 0s?', np.mean(cosine_similarity(G1_fair, G3_fair)))
    print('How different are red 1s from red 0s?', np.mean(cosine_similarity(G2_fair, G4_fair)))

    N = 200
    X = np.concatenate((G1[:N], G2[:N], G3[:N], G4[:N]), axis=0)
    y = np.array([0] * N + [1] * N + [0] * N + [1] * N)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('How good are the original embeddings at separating colors?', accuracy)

    y_number = np.array([0] * (N * 2) + [1] * (N * 2))
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y_number, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('How good are the original embeddings at separating numbers?', accuracy)

    N = 200
    X = np.concatenate((G1_fair[:N], G2_fair[:N], G3_fair[:N], G4_fair[:N]), axis=0)
    y = np.array([0] * N + [1] * N + [0] * N + [1] * N)
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('How good are our fair embeddings at separating colors?', accuracy)

    y_number = np.array([0] * (N * 2) + [1] * (N * 2))
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y_number, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('How good are our fair embeddings at separating numbers?', accuracy)
    exit()

    G = scale_G(np.concatenate((G1, G2), axis=0))
    A = compute_covariance(G)
    v, e = get_real_eig(A)

    mean_color_direction = e[:, -1].reshape(-1, 1).T

    print('How different are muted blues from blues?', cosine_similarity(np.mean(G1 - mean_color_direction, axis=0).reshape(1, -1), np.mean(G1, axis=0).reshape(1, -1)))


    G1 -= mean_color_direction
    G2 -= mean_color_direction

    similarity_matrix = cosine_similarity(G1, G2)
    average_similarity = np.mean(similarity_matrix)
    print(average_similarity)

    exit()

    G_prime = scale_G(np.concatenate((G1, G2), axis=0))
    A = compute_covariance(G_prime)
    v, e = get_real_eig(A)

    m = 3
    color_direction = e[:, -m]
    mean_color_direction = (np.mean(G2, axis=0) - np.mean(G1, axis=0)).reshape(1, -1)
    print(cosine_similarity(color_direction.reshape(-1, 1).T, mean_color_direction))

    G1 -= mean_color_direction
    G2 -= mean_color_direction

    P = np.eye(84) - mean_color_direction.T @ mean_color_direction
    return P



def get_fair_subspace_MANUAL_2_GROUPS(majority_group, minority_group, G, subclasses):
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
    # print('std before:', std_dev)

    tolerance = std_dev / 10
    keep_idx = []
    keep_diffs = []
    for i in range(len(variances_1)):
        if abs(variances_1[i] - variances_2[i]) < tolerance: 
            keep_idx.append(i)
            keep_diffs.append(abs(variances_1[i] - variances_2[i]))
    
    print('directions retained:', len(keep_idx))

    # print('std after:', np.std(keep_diffs))

    D = len(eigenvectors[0])

    P = np.zeros((D, D), dtype='float')
    for idx in keep_idx:
        u_i = eigenvectors[:, idx].reshape((D, 1))
        P += (u_i @ u_i.T)

    # U = np.concatenate([eigenvectors[:, idx].reshape(D, -1) for idx in keep_idx], axis=1)
    # G.dot(P) N x D
    # G.dot(U) N x k

    span = eigenvectors[:, keep_idx[0]].reshape(-1, 1)
    for i in range(1, len(keep_idx)):
        span = np.concatenate((span, eigenvectors[:, keep_idx[i]].reshape(-1, 1)), axis=1)

    return P, span

def get_fair_subspace_MANUAL_2_GROUPS_COSINE(majority_group, minority_group, G, subclasses):
    G_1 = G[subclasses == majority_group] # M x D
    G_2 = G[subclasses == minority_group] # m x D

    A_1 = compute_covariance(G)
    eigenvalues_1, eigenvectors_1 = get_real_eig(A_1)

    A_2 = compute_covariance(G_2)
    eigenvalues_2, eigenvectors_2 = get_real_eig(A_2)

    similarity_matrix = cosine_similarity(eigenvectors_1.T, eigenvectors_2.T)
    similarities = np.diag(similarity_matrix)

    threshold = np.mean(similarities)

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

    return P, average_eigenvectors



    
    




