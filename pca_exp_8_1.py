import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd
from scipy.linalg import null_space

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from fair_k_pca import get_fair_subspace, get_fair_subspace_MANUAL_2_GROUPS, get_fair_subspace_MANUAL_N_GROUPS, get_fair_subspace_MANUAL_2_GROUPS_COSINE

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

def intra_group_distance(embeddings, metric='euclidean'):
    pairwise_distances = pdist(embeddings, metric=metric)
    return np.mean(pairwise_distances)

def inter_group_distance(embeddings1, embeddings2, metric='euclidean'):
    pairwise_distances = cdist(embeddings1, embeddings2, metric)
    return np.mean(pairwise_distances)

def scale_G(G):
    scaler = StandardScaler()
    G = scaler.fit_transform(G)
    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise
    return G

def subspace_transfer():
    _, embeddings, subclasses, _, _, _ = unpack_data('train_14_epoch')

    group1 = [6, 9]
    group2 = [1, 4]

    G1 = embeddings[np.isin(subclasses, group1)]
    G1 = scale_G(G1)
    G2 = embeddings[np.isin(subclasses, group2)]
    G2 = scale_G(G2)

    A1 = compute_covariance(G1)
    A2 = compute_covariance(G2)

    k = 50

    v1, e1 = get_real_eig(A1)
    pca1 = e1[:, -k:].reshape(-1, k)

    v2, e2 = get_real_eig(A2)
    pca2 = e2[:, -k:].reshape(-1, k)


    u_space, pca_fair = get_fair_subspace(G1, subclasses[np.isin(subclasses, group1)], k, group1)

    obj_pca1 = np.trace((pca1 @ pca1.T) @ A1.T) 
    obj_fpca1 = np.trace((pca_fair @ pca_fair.T) @ A1.T) 
    obj_pca2 = np.trace((pca2 @ pca2.T) @ A2.T) 
    obj_fpca2 = np.trace((pca_fair @ pca_fair.T) @ A2.T) 
    print(f'50PCA  Objective on Group 1: {obj_pca1:<50.4f}')
    print(f'F50PCA Objective on Group 1: {obj_fpca1:<50.4f}')
    print(f'50PCA  Objective on Group 2: {obj_pca2:<50.4f}')
    print(f'F50PCA Objective on Group 2: {obj_fpca2:<50.4f}')

    proj1 = scale_G(G1 @ pca_fair)
    proj2 = scale_G(G2 @ pca_fair)
    A_proj1 = compute_covariance(proj1)
    A_proj2 = compute_covariance(proj2)
    v1, e1 = get_real_eig(A_proj1)
    v2, e2 = get_real_eig(A_proj2)
    intersection = e1[:, -k:].T @ e2[:, -k:]
    canonical_correlations = np.sort(svd(intersection, compute_uv=False))
    print('Top 10 Canonical Correlations:', canonical_correlations[-10:])

    G_blue1 = scale_G(embeddings[subclasses == 6])
    G_red1 = scale_G(embeddings[subclasses == 9])
    G_blue2 = scale_G(embeddings[subclasses == 1])
    G_red2 = scale_G(embeddings[subclasses == 4])

    G1_ordered = scale_G(np.concatenate((G_blue1, G_red1), axis=0))
    A1_ordered = compute_covariance(G1_ordered)
    v, e = get_real_eig(A1_ordered)
    
    data = {
        'color': np.array([0] * len(G_blue1) + [1] * len(G_red1))
    }
    df = pd.DataFrame(data)
    pca_df = pd.DataFrame(data=G1_ordered @ e[:, -2:], columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('2PCA of Blue and Red 0s Embeddings')
    plt.savefig('pca_exp_8_1/2PCA_blue_red_0s_experiment.png')
    plt.clf()

    data = {
        'color': np.array([0] * len(G_blue1) + [1] * len(G_red1))
    }
    df = pd.DataFrame(data)
    pca_df = pd.DataFrame(data=(G1_ordered @ pca_fair[:, -2:]), columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('2FPCA of Blue and Red 0s Embeddings')
    plt.savefig('pca_exp_8_1/2FPCA_blue_red_0s_experiment.png')
    plt.clf()

    #############################################################################################

    G2_ordered = scale_G(np.concatenate((G_blue2, G_red2), axis=0))
    A2_ordered = compute_covariance(G2_ordered)
    v, e = get_real_eig(A2_ordered)

    data = {
        'color': np.array([0] * len(G_blue2) + [1] * len(G_red2))
    }
    df = pd.DataFrame(data)
    pca_df = pd.DataFrame(data=G2_ordered @ e[:, -2:], columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('2PCA of Blue and Red 1s Embeddings')
    plt.savefig('pca_exp_8_1/2PCA_blue_red_1s_experiment.png')
    plt.clf()

    data = {
        'color': np.array([0] * len(G_blue2) + [1] * len(G_red2))
    }
    df = pd.DataFrame(data)
    pca_df = pd.DataFrame(data=(G2_ordered @ pca_fair[:, -2:]), columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('2FPCA of Blue and Red 1s Embeddings')
    plt.savefig('pca_exp_8_1/2FPCA_blue_red_1s_experiment.png')
    plt.clf()

def inferring_color_subspace():
    # pca on all blue digits
    # fpca on groups (blue 0s, blue 1s)

    _, embeddings, subclasses, _, _, _ = unpack_data('train_14_epoch')

    blue_groups = [1, 6]
    G = embeddings[np.isin(subclasses, blue_groups)]
    scaler = StandardScaler()
    G = scaler.fit_transform(G)

    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    for k in [1, 10, 20, 30, 40, 50, 60, 70]:
        A = compute_covariance(G)
        v, e = get_real_eig(A)
        pca = e[:, -k:].reshape(-1, k)

        u_space, y_opt, pca_fair = get_fair_subspace(G, subclasses[np.isin(subclasses, blue_groups)], k, blue_groups)

        obj_fpca = np.trace((pca_fair @ pca_fair.T) @ A.T) 
        obj_pca1 = np.trace((pca @ pca.T) @ A.T)

        print('Full FPCA Objective Val:', np.trace((y_opt @ y_opt.T) @ A.T))
        print('FPCA      Objective Val:', obj_fpca)
        print(f'PCA{k}   Objective Val:', obj_pca1)

        if k == 1:
            print('Cosine Similarity', cosine_similarity(pca.T, pca_fair.T))
        elif k > 1:
            intersection = pca.T @ pca_fair
            canonical_correlations = np.sort(svd(intersection, compute_uv=False))
            print('Top 10 Canonical Correlations:', canonical_correlations[-40:])


        print('===========================================================================================')

def visualization_of_inferred_color_subspace():
    _, embeddings, subclasses, _, _, _ = unpack_data('train_14_epoch')

    group1 = 6
    group2 = 1

    G = np.concatenate((embeddings[subclasses == group1], embeddings[subclasses == group2]), axis=0)
    scaler = StandardScaler()
    G = scaler.fit_transform(G)

    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise

    k = 50

    A = compute_covariance(G)
    v, e = get_real_eig(A)
    pca = e[:, -k:].reshape(-1, k)

    ordered_subclasses = np.array([0] * len(embeddings[subclasses == group1]) + [1] * len(embeddings[subclasses == group2]))
    u_space, pca_fair = get_fair_subspace(G, ordered_subclasses, k, [0, 1])

    intersection = pca.T @ pca_fair
    canonical_correlations = np.sort(svd(intersection, compute_uv=False))
    print('Top 10 Canonical Correlations:', canonical_correlations[-10:])
    
    data = {
        'color': ordered_subclasses
    }
    df = pd.DataFrame(data)
    project_pca = G @ pca[:, -2:]
    pca_df = pd.DataFrame(data=project_pca, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('2PCA of Blue Embeddings')
    plt.savefig('pca_exp_8_1/2PCA_blue_experiment.png')
    plt.clf()

    data = {
        'color': ordered_subclasses
    }
    df = pd.DataFrame(data)

    project_pca = G @ pca_fair[:, -2:]
    pca_df = pd.DataFrame(data=project_pca, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[['color']]], axis=1)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='color', data=pca_df, s=100)
    plt.title('F2PCA of Blue Embeddings')
    plt.savefig('pca_exp_8_1/F2PCA_blue_experiment.png')


def intersect_subspaces(embeddings, subclasses, k):
    u1, _, _ = get_fair_subspace(scale_G(embeddings), subclasses, k, [0, 1, 0, 1, 0, 1])
    u2, _, _ = get_fair_subspace(scale_G(embeddings), subclasses, k, [2, 3])
    # u = np.vstack([u1, u2])
    # u = null_space(u)
    u = u1.T @ u2
    print('Rank of Intersection:', np.linalg.matrix_rank(u))
    return u

    
def out_of_dist_generalization():
    _, embeddings, subclasses, labels, _, _ = unpack_data('CMNIST_meta_train_15_epoch')

    color1 = [1, 3]
    color2 = [0, 2]

    ############################################
    # blue1_s1 = embeddings[subclasses == 0][0]
    # blue1_s2 = embeddings[subclasses == 0][1]
    # print('How close are blue1s to blue1s?', cosine_similarity(blue1_s1.reshape(1, -1), blue1_s2.reshape(1, -1)))
    # blue0_s1 = embeddings[subclasses == 2][0]
    # blue0_s2 = embeddings[subclasses == 2][1]
    # print('How close are blue0s to blue0s?', cosine_similarity(blue0_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are blue1s to blue0s?', cosine_similarity(blue1_s1.reshape(1, -1), blue0_s1.reshape(1, -1)))

    # red1_s1 = embeddings[subclasses == 1][0]
    # red1_s2 = embeddings[subclasses == 1][1]
    # red0_s1 = embeddings[subclasses == 3][0]
    # red0_s2 = embeddings[subclasses == 3][1]
    # print('How close are red1s to red1s?', cosine_similarity(red1_s1.reshape(1, -1), red1_s2.reshape(1, -1)))
    # print('How close are red0s to red0s?', cosine_similarity(red0_s1.reshape(1, -1), red0_s2.reshape(1, -1)))
    # print('How close are red0s to red1s?', cosine_similarity(red0_s1.reshape(1, -1), red1_s2.reshape(1, -1)))
    # print('How close are red1s to blue1s?', cosine_similarity(red1_s1.reshape(1, -1), blue1_s1.reshape(1, -1)))
    # print('How close are red1s to blue0s?', cosine_similarity(red1_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are red0s to blue0s?', cosine_similarity(red0_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are red0s to blue1s?', cosine_similarity(red0_s1.reshape(1, -1), blue1_s2.reshape(1, -1)))
    ############################################

    def run_classifier(X, y, question):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(question, accuracy)
    
    X = scale_G(np.concatenate((embeddings[subclasses == 1][:500], embeddings[subclasses == 3][:500], embeddings[subclasses == 0][:500], embeddings[subclasses == 2][:500]), axis=0))
    y_color = np.array([1] * 1000 + [0] * 1000)
    y_number = np.array([1] * 500 + [0] * 500 + [1] * 500 + [0] * 500)
    run_classifier(X, y_color, "How well can we separate colors with the original embeddings?")
    run_classifier(X, y_number, "How well can we separate numbers with the original embeddings?")

    
    def compute_pca(k):
        A = compute_covariance(X)
        v, e = get_real_eig(A)
        pca = e[:, -k:].reshape(-1, k)
        P = pca @ pca.T
        X_pca = X @ P
        run_classifier(X_pca, y_color, f"How well can we separate colors with the top {k} components of the original embeddings?")
        run_classifier(X_pca, y_number, f"How well can we separate numbers with the top {k} components of the original embeddings?")

    compute_pca(1)
    compute_pca(2)
    compute_pca(3)


    for k in [10, 20, 30, 40, 50, 60, 70]:
        # u_space, _, pca_fair = get_fair_subspace(scale_G(embeddings), labels, k, [0, 1])
        # u_space, pca_fair = get_fair_subspace_MANUAL_2_GROUPS(1, 0, scale_G(embeddings), labels)
        u_space = intersect_subspaces(embeddings, subclasses, k)
        X_proj = X @ u_space
        run_classifier(X_proj, y_color, f"How well can we separate colors with F{k}PCA?")
        run_classifier(X_proj, y_number, f"How well can we separate numbers with F{k}PCA?")
        print('====================================================================')


def compare_algo_to_manual_subspace():
    _, embeddings, subclasses, labels, _, _ = unpack_data('CMNIST_meta_train_15_epoch')

    G = scale_G(embeddings)
    A = compute_covariance(G)

    print('getting fair subspace with sdp')
    # u1, _, s1 = get_fair_subspace(G, labels, 20, [0, 1])
    u = intersect_subspaces(embeddings, subclasses, 20)
    Au = compute_covariance(scale_G(u))
    v, e = get_real_eig(Au)
    s1 = e[:, -20:]
    print('SDP Objective Val:', np.trace((s1 @ s1.T) @ A.T) )
    print('getting manual fair subspace with stddev comparison')
    u2, s2 = get_fair_subspace_MANUAL_2_GROUPS(0, 1, G, labels)
    print('Manual Stddev Objective Val:', np.trace((s2 @ s2.T) @ A.T) )
    print('getting manual fair subspace with cosine comparison')
    u3, s3 = get_fair_subspace_MANUAL_2_GROUPS_COSINE(0, 1, G, labels)
    print('Manual Cosine Objective Val:', np.trace((s3 @ s3.T) @ A.T) )

    intersection = s2.T @ s3
    canonical_correlations = np.sort(svd(intersection, compute_uv=False))
    print('Manual v Manual Top 10 Canonical Correlations:', canonical_correlations[-10:])

    intersection = s1.T @ s2
    canonical_correlations = np.sort(svd(intersection, compute_uv=False))
    print('SDP v Manual Top 10 Canonical Correlations:', canonical_correlations[-10:])

    intersection = s1.T @ s3
    canonical_correlations = np.sort(svd(intersection, compute_uv=False))
    print('SDP v Manual Cosine Top 10 Canonical Correlations:', canonical_correlations[-10:])



def comparing_embeddings():
    logits, embeddings, subclasses, labels, _, _ = unpack_data('CMNIST_meta_train_15_epoch')
    predictions = np.argmax(logits, axis=1)

    wrong_blue1s = embeddings[(subclasses == 0) & (predictions != labels)]
    wrong_blue0s = embeddings[(subclasses == 2) & (predictions != labels)]

    wrong_red1s = embeddings[(subclasses == 1) & (predictions != labels)]
    wrong_red0s = embeddings[(subclasses == 3) & (predictions != labels)]

    correct_blue1s = embeddings[(subclasses == 0) & (predictions == labels)]
    correct_blue0s = embeddings[(subclasses == 2) & (predictions == labels)]
    
    correct_red1s = embeddings[(subclasses == 1) & (predictions == labels)]
    correct_red0s = embeddings[(subclasses == 3) & (predictions == labels)]

    # def avg_cosine_sim(a, b):
    #     cosine_sims = []
    #     for v1 in a:
    #         for v2 in b:
    #             print('hi')
    #             cosine_sims.append(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])
    #     return np.mean(cosine_sims)

    def compare(a, b, exclude_diagonal=False):
        sim_matrix = cosine_similarity(a, b)
        if exclude_diagonal:
            mask = np.eye(len(a), len(b), dtype=bool)
            sim_matrix = sim_matrix[~mask]
        return [np.mean(sim_matrix), inter_group_distance(a, b, 'euclidean')]

    print('How close are wrong blue1s to wrong blue1s?', compare(wrong_blue1s, wrong_blue1s, True)) # should be close
    print('How close are wrong blue1s to correct blue1s?', compare(wrong_blue1s, correct_blue1s)) # should be far
    print('How close are wrong blue1s to wrong blue0s?', compare(wrong_blue1s, wrong_blue0s)) # idk
    print('How close are wrong blue1s to correct blue0s?', compare(wrong_blue1s, correct_blue0s)) # should be close
    print('How close are wrong blue1s to wrong red1s?', compare(wrong_blue1s, wrong_red1s)) # should be far
    print('How close are wrong blue1s to correct red1s?', compare(wrong_blue1s, correct_red1s)) # should be far
    print('How close are wrong blue1s to wrong red0s?', compare(wrong_blue1s, wrong_red0s)) # should be far
    print('How close are wrong blue1s to right red0s?', compare(wrong_blue1s, correct_red0s)) # should be far
    print()
    print('How close are wrong blue0s to wrong blue0s?', compare(wrong_blue0s, wrong_blue0s, True)) # should be close
    print('How close are wrong blue0s to correct blue0s?', compare(wrong_blue0s, correct_blue0s)) # should be far
    print('How close are wrong blue0s to correct blue1s?', compare(wrong_blue0s, correct_blue1s))
    print('How close are wrong blue0s to wrong red1s?', compare(wrong_blue0s, wrong_red1s)) # should be far
    print('How close are wrong blue0s to correct red1s?', compare(wrong_blue0s, correct_red1s)) # should be far
    print('How close are wrong blue0s to wrong red0s?', compare(wrong_blue0s, wrong_red0s)) # should be far
    print('How close are wrong blue0s to right red0s?', compare(wrong_blue0s, correct_red0s)) # should be far
    print()
    print('How close are correct blue1s to correct blue1s?', compare(correct_blue1s, correct_blue1s, True)) # should be close
    print('How close are correct blue1s to correct blue0s?', compare(correct_blue1s, correct_blue0s)) # idk
    print('How close are correct blue1s to wrong red1s?', compare(correct_blue1s, wrong_red1s)) # should be far
    print('How close are correct blue1s to correct red1s?', compare(correct_blue1s, correct_red1s)) # should be far
    print('How close are correct blue1s to wrong red0s?', compare(correct_blue1s, wrong_red0s)) # should be far
    print('How close are correct blue1s to right red0s?', compare(correct_blue1s, correct_red0s)) # should be far
    print()
    print('How close are correct blue0s to correct blue0s?', compare(correct_blue0s, correct_blue0s, True)) # should be close
    print('How close are correct blue0s to wrong red1s?', compare(correct_blue0s, wrong_red1s)) # should be far
    print('How close are correct blue0s to correct red1s?', compare(correct_blue0s, correct_red1s)) # should be far
    print('How close are correct blue0s to wrong red0s?', compare(correct_blue0s, wrong_red0s)) # should be far
    print('How close are correct blue0s to right red0s?', compare(correct_blue0s, correct_red0s)) # should be far



    # blue1_s1 = embeddings[subclasses == 0][0]
    # blue1_s2 = embeddings[subclasses == 0][1]
    # print('How close are blue1s to blue1s?', cosine_similarity(blue1_s1.reshape(1, -1), blue1_s2.reshape(1, -1)))
    # blue0_s1 = embeddings[subclasses == 2][0]
    # blue0_s2 = embeddings[subclasses == 2][1]
    # print('How close are blue0s to blue0s?', cosine_similarity(blue0_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are blue1s to blue0s?', cosine_similarity(blue1_s1.reshape(1, -1), blue0_s1.reshape(1, -1)))

    # red1_s1 = embeddings[subclasses == 1][0]
    # red1_s2 = embeddings[subclasses == 1][1]
    # red0_s1 = embeddings[subclasses == 3][0]
    # red0_s2 = embeddings[subclasses == 3][1]
    # print('How close are red1s to red1s?', cosine_similarity(red1_s1.reshape(1, -1), red1_s2.reshape(1, -1)))
    # print('How close are red0s to red0s?', cosine_similarity(red0_s1.reshape(1, -1), red0_s2.reshape(1, -1)))
    # print('How close are red0s to red1s?', cosine_similarity(red0_s1.reshape(1, -1), red1_s2.reshape(1, -1)))
    # print('How close are red1s to blue1s?', cosine_similarity(red1_s1.reshape(1, -1), blue1_s1.reshape(1, -1)))
    # print('How close are red1s to blue0s?', cosine_similarity(red1_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are red0s to blue0s?', cosine_similarity(red0_s1.reshape(1, -1), blue0_s2.reshape(1, -1)))
    # print('How close are red0s to blue1s?', cosine_similarity(red0_s1.reshape(1, -1), blue1_s2.reshape(1, -1)))


# inferring_color_subspace()
# visualization_of_inferred_color_subspace()
# subspace_transfer()
# out_of_dist_generalization()
# compare_algo_to_manual_subspace()
comparing_embeddings()

'''
sub1 = spanX, sub2 = spanY
similarity(sub1, sub2) = eigenvalues of (X.T @ Y)
'''

'''
fpca on blue 1s and red 1s
get complement of the fair 1s subspace
do same for 2s
intersect them

how close is the gap:
1. top k eigenvalues
2. compute the dual take top k smth dual
3. manually pick k directions
'''


