import numpy as np

def whiten(embeddings):
    G = embeddings
    G_bar = G.mean(axix=0, keepdims=True)
    G_centered = G - G_bar
    cov_matrix = np.cov(G_centered, rowvar=False)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    W = np.linalg.cholesky(cov_matrix_inv)
    return W

