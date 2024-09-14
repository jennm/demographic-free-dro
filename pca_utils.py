import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def unpack_data(part):
    train_npz = np.load(f'embeddings/{part}.npz')
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

def scale_G(G):
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    G = G / norms
    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise
    return G

def compare_embeddings(a, b, exclude_diagonal=False):
    sim_matrix = cosine_similarity(a, b)
    if exclude_diagonal:
        mask = np.eye(len(a), len(b), dtype=bool)
        sim_matrix = sim_matrix[~mask]

    distances = []

    for _a in a:
        for _b in b:
            distance = np.linalg.norm(_a - _b)
            distances.append(distance)
    average_distance = np.mean(distances)

    return {'Cosine Similarity:': np.mean(sim_matrix), 'Euclidean Distance:': average_distance}