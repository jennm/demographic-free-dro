from pca_utils import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

logits, embeddings, subclasses, labels, _, losses = unpack_data('CelebA_meta_train_1_epoch')
test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('CelebA_meta_test_12_epoch')

# embeddings = np.concatenate((embeddings, losses.reshape(-1, 1)), axis=1)
embeddings = scale_G(embeddings)
test_embeddings = scale_G(test_embeddings)

subclasses[subclasses == 0] = 10
subclasses[subclasses == 1] = 12
subclasses[subclasses == 2] = 11
subclasses[subclasses == 3] = 13
subclasses[subclasses == 10] = 0 # female brunettes
subclasses[subclasses == 11] = 1 # female blondes
subclasses[subclasses == 12] = 2 # male brunettes
subclasses[subclasses == 13] = 3 # male blondes

test_subclasses[test_subclasses == 0] = 10
test_subclasses[test_subclasses == 1] = 12
test_subclasses[test_subclasses == 2] = 11
test_subclasses[test_subclasses == 3] = 13
test_subclasses[test_subclasses == 10] = 0
test_subclasses[test_subclasses == 11] = 1
test_subclasses[test_subclasses == 12] = 2
test_subclasses[test_subclasses == 13] = 3

predictions = np.argmax(logits, axis=1)
misclassified = predictions != labels

# print(len(subclasses[subclasses == 0]))
# print(len(subclasses[subclasses == 1]))
# print(len(subclasses[subclasses == 2]))
# print(len(subclasses[subclasses == 3]))

'''
Male => brunette (really strong correlation)
Female => blonde (mid correlation)
asymmetric correlation
'''


female_brunettes = embeddings[(subclasses == 0)]# & (misclassified == 0)]
female_blondes = embeddings[(subclasses == 1)]# & (misclassified == 0)]
male_brunettes = embeddings[(subclasses == 2)]# & (misclassified == 0)]
male_blondes = embeddings[(subclasses == 3)]# & (misclassified == 1)]

N = 200
X_train = np.concatenate((female_brunettes[:N], female_blondes[:N], male_brunettes[:N], male_blondes[:N]), axis=0)
X_test = test_embeddings
y_train_color = np.array([1] * N + [0] * N + [1] * N + [0] * N)
y_train_gender = np.array([1] * N * 2 + [0] * N * 2)
y_test_color = np.isin(test_subclasses, [0, 2])
y_test_gender = np.isin(test_subclasses, [0, 1])

y_train_group = np.array([0] * N + [1] * N + [2] * N + [3] * N)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(X_train)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_train_group, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Group')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of 84-Dimensional Embeddings')
plt.savefig('8_23.png')


'''
color_cls
number_cls

w_color
w_number

u = w_color - w_number

project out u
'''

'''
Why are male blondes so close to female brunettes?
male -> brunette
predictive signal for a male blonde -> brunette
"it thinks any blonde male is a female brunette"

Why are male blondes farthest apart from female blondes?
male -> brunettes
"it thinks any blonde male should be a brunette"


'''