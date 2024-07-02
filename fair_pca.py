import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

def unpack_data(part):
    train_npz = np.load(f'cmnist_meta_{part}.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    losses = train_npz['loss']
    
    return predictions, embeddings, subclasses, labels, data_idx, losses


def experiment():
    # load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')

    # row_means = np.mean(embeddings, axis=1, keepdims=True)
    # row_stds = np.std(embeddings, axis=1, keepdims=True)
    # embeddings = (embeddings - row_means) / row_stds

    # arr_mean = np.mean(losses)
    # arr_std = np.std(losses)
    # losses = (losses - arr_mean) / arr_std

    # # pca = PCA(n_components=42)
    # # embeddings = pca.fit_transform(embeddings)
    # # print('Reduced Embedding Shape:', embeddings.shape)

    # embeddings = np.hstack((embeddings, losses.reshape(-1, 1)))

    majority_subclass = 0
    minority_subclass = 3

    sample_size = 100

    G_1 = embeddings[subclasses == majority_subclass][:sample_size]
    G_2 = embeddings[subclasses == minority_subclass][:sample_size]
    G = np.concatenate((G_1, G_2), axis=0)

    G_1_bar = G_1.mean(axis=1, keepdims=True)
    G_2_bar = G_2.mean(axis=1, keepdims=True)
    G_bar = G.mean(axis=1, keepdims=True)

    G_1_centered = G_1 - G_1_bar
    G_2_centered = G_2 - G_2_bar
    G_centered = G - G_bar

    # covariance matrices: tells us how much each pair of dimensions change together
    # if 2 pairs of dimensions change together a lot (i.e. covariance is high), it means there's some strong signal in their direction
    A_1 = G_1_centered.T @ G_1_centered
    A_2 = G_2_centered.T @ G_2_centered
    A = G_centered.T @ G_centered

    D = embeddings.shape[-1]
    frac = 1.0
    n_components = 2 # int(D * frac)

    pca = PCA(n_components=n_components)

    '''
    each column corresponds to a different principal component
    the first principal component corresponds to the first column
    the first principal component gives us the linear combination of variables that maximize the variance
    the Nth gives us the Nth best linear combination
    each element in each column gives us the "importance" of the corresponding dimension for the Nth principal component

    direction with highest variance in a particular digit class should pick up the color features (spurious = red)
    the dimensions that have the most weight for this direction should be the features that correspond to color
    '''

    u = pca.fit_transform(A) # 84 x 2

    # projection_A1 = np.dot(A_1, u)
    # projection_A2 = np.dot(A_2, u)
    # projection_A = np.dot(A, u)

    # variance_explained_A1 = np.var(projection_A1)
    # variance_explained_A2 = np.var(projection_A2)
    # variance_explained_A = np.var(projection_A)

    # print('u on A1:', variance_explained_A1)
    # print('u on A2:', variance_explained_A2)
    # print('u on A:', variance_explained_A)

    print('G var exp', u.shape, G.shape)
    print('G1 var exp', np.var(np.dot(G_1, u)))
    print('G2 var exp', np.var(np.dot(G_2, u)))

    # visualization
    data = {
        'color': ['red'] * 100 + ['blue'] * 100,
        'shape': ['circle'] * 100 + ['square'] * 100
    }
    df = pd.DataFrame(data)

    projected_data = G_centered.dot(u)

    pca_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color', 'shape']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', style='shape', data=pca_df, s=100)
    plt.title('PCA of Shape and Color Data')
    plt.savefig('PCA_experiment.png')

def experiment2():
# load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')

    majority_subclass = 0
    minority_subclass = 3

    sample_size = 100

    G_1 = embeddings[subclasses == majority_subclass][:sample_size]
    G_2 = embeddings[subclasses == minority_subclass][:sample_size]
    G = np.concatenate((G_1, G_2), axis=0)

    G_1_bar = G_1.mean(axis=1, keepdims=True)
    G_2_bar = G_2.mean(axis=1, keepdims=True)
    G_bar = G.mean(axis=1, keepdims=True)

    G_1_centered = G_1 - G_1_bar
    G_2_centered = G_2 - G_2_bar
    G_centered = G - G_bar

    A = G_centered.T @ G_centered

    print(np.all(np.isreal(A)))
    print(np.allclose(A, A.T))

    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    if np.all(np.abs(np.imag(eigenvalues)) < 1e-10):
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    variances_1 = []
    variances_2 = []
    variances_diffs = []

    for i in range(len(eigenvectors[0])):
        eigenvector = eigenvectors[:, i]
        variances_1.append(np.var(np.dot(G_1_centered, eigenvector)))
        variances_2.append(np.var(np.dot(G_2_centered, eigenvector)))
        variances_diffs.append(abs(variances_1[-1] - variances_2[-1]))

    x = list(range(len(eigenvectors)))

    std_dev = np.std(variances_diffs)
    print('std before:', std_dev)

    tolerance = std_dev / 10

    keep_idx = []
    keep_diffs = []
    for i in range(len(variances_1)):
        if abs(variances_1[i] - variances_2[i]) < tolerance: 
            keep_idx.append(i)
            keep_diffs.append(abs(variances_1[i] - variances_2[i]))
    
    print(len(keep_idx))
    print('std after:', np.std(keep_diffs))

    print(eigenvectors.shape)

    D = len(eigenvectors[0])

    P = np.zeros((D, D), dtype='float')
    for idx in keep_idx:
        u_i = eigenvectors[:, i].reshape((D, 1))
        P += (u_i @ u_i.T)

    data = {
        'color': ['red'] * 100 + ['blue'] * 100,
    }
    df = pd.DataFrame(data)

    projected_data = G_centered.dot(P)
    print('projected_data.shape', projected_data.shape)
    print(projected_data[:, :2].shape)

    G_proj_bar = projected_data.mean(axis=1, keepdims=True)
    G_proj_centered = projected_data - G_proj_bar

    A_proj = G_proj_centered.T @ G_proj_centered
    pca = PCA(n_components=2)
    u_proj = pca.fit_transform(A_proj)

    viz_projected_data = G_proj_centered.dot(u_proj)

    pca_df = pd.DataFrame(data=viz_projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('PCA of Shape and Color Data')
    plt.savefig('PCA_experiment2.png')

# experiment2()

def experiment3():
    # load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')

    # row_means = np.mean(embeddings, axis=1, keepdims=True)
    # row_stds = np.std(embeddings, axis=1, keepdims=True)
    # embeddings = (embeddings - row_means) / row_stds

    # arr_mean = np.mean(losses)
    # arr_std = np.std(losses)
    # losses = (losses - arr_mean) / arr_std

    # # pca = PCA(n_components=42)
    # # embeddings = pca.fit_transform(embeddings)
    # # print('Reduced Embedding Shape:', embeddings.shape)

    # embeddings = np.hstack((embeddings, losses.reshape(-1, 1)))

    G_1 = embeddings[subclasses == majority_subclass]
    G_2 = embeddings[subclasses == minority_subclass]
    G = np.concatenate((G_1, G_2), axis=0)

    G_1_bar = G_1.mean(axis=1, keepdims=True)
    G_2_bar = G_2.mean(axis=1, keepdims=True)
    G_bar = G.mean(axis=1, keepdims=True)

    G_1_centered = G_1 - G_1_bar
    G_2_centered = G_2 - G_2_bar
    G_centered = G - G_bar

    # covariance matrices: tells us how much each pair of dimensions change together
    # if 2 pairs of dimensions change together a lot (i.e. covariance is high), it means there's some strong signal in their direction
    A_1 = G_1_centered.T @ G_1_centered
    A_2 = G_2_centered.T @ G_2_centered
    A = G_centered.T @ G_centered

    D = embeddings.shape[-1]
    frac = 1.0
    n_components = 2 # int(D * frac)

    pca = PCA(n_components=n_components)
    u = pca.fit_transform(A) # 84 x 2

    print('G var exp', u.shape, G.shape)
    print('G1 var exp', np.var(np.dot(G_1, u)))
    print('G2 var exp', np.var(np.dot(G_2, u)))

    # visualization
    data = {
        'color': ['red'] * 100 + ['blue'] * 100,
        'shape': ['circle'] * 100 + ['square'] * 100
    }
    df = pd.DataFrame(data)

    projected_data = G_centered.dot(u)

    pca_df = pd.DataFrame(data=projected_data, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color', 'shape']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', style='shape', data=pca_df, s=100)
    plt.title('PCA of Shape and Color Data')
    plt.savefig('PCA_experiment.png')

experiment3()

'''
to remove spurious correlation:
- project embeddings onto orthogonal complement
'''

'''
stuff i don't get:
- how does PCA actually work
- why is a dot product a projection
- how is covariance matrix calculated

can you view a principal component as a hyperplane?
'''