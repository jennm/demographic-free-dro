import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from itertools import chain
import random

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from retrain_pca import LinearModel, train, evaluate
from pca_utils import *
from dp_sgd import DPSGD

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random.seed(42)


'''
At epoch 1, how similar are blue1s from red1s? {'Cosine Similarity:': 0.21729694817434464, 'Euclidean Distance:': 1.2509849364688355}
At epoch 16, how similar are blue1s from red1s? {'Cosine Similarity:': 0.2599164911628531, 'Euclidean Distance:': 1.2147283035721728}
At epoch 1, how similar are blue1s to blue0s? {'Cosine Similarity:': 0.9849205075328626, 'Euclidean Distance:': 0.1658383567031586}
At epoch 16, how similar are blue1s to blue0s? {'Cosine Similarity:': 0.7316833282467863, 'Euclidean Distance:': 0.720206375246414}
How similar are blue1s from epoch 1 to blue1s from epoch 16? {'Cosine Similarity:': 0.7154569816774586, 'Euclidean Distance:': 0.7463178559565041}
After masking out epoch 1, how similar are blue0s and blue1s? {'Cosine Similarity:': 0.38653689582613515, 'Euclidean Distance:': 0.7080502784518566}
After masking out epoch 1, how similar are blue0s and red0s? {'Cosine Similarity:': 0.03455711858742066, 'Euclidean Distance:': 0.8015356266986267}
After masking out epoch 1, how similar are red0s and red1s? {'Cosine Similarity:': 0.31561228079475373, 'Euclidean Distance:': 0.6585733367459837}
After masking out epoch 1, how  similar are red1s and blue1s? {'Cosine Similarity:': 0.015708844682785966, 'Euclidean Distance:': 0.8697612779650784}

blue0s and blue1s are both aligned well in the same the direction AND are also really close together
After subtracting out the embeddings from epoch 1, everything is much closer together but the directions are all misaligned
'''
def compare_embeddings_across_epochs():
    logits_1, embeddings_1, subclasses_1, labels_1, _, _ = unpack_data('ColoredMNIST_HARD_meta_train_1_epoch')
    logits_16, embeddings_16, subclasses_16, labels_16, _, _ = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')

    predictions_1 = np.argmax(logits_1, axis=1)
    predictions_16 = np.argmax(logits_16, axis=1)

    misclassified_1 = predictions_1 != labels_1
    misclassified_16 = predictions_16 != labels_16

    embeddings_1 = scale_G(embeddings_1)
    embeddings_16 = scale_G(embeddings_16)

    blue1s_1 = embeddings_1[(subclasses_1 == 0) & (misclassified_1 == 1)]
    red1s_1 = embeddings_1[(subclasses_1 == 1) & (misclassified_1 == 0)]
    blue1s_16 = embeddings_16[(subclasses_16 == 0) & (misclassified_16 == 1)]
    red1s_16 = embeddings_16[(subclasses_16 == 1) & (misclassified_16 == 0)]

    blue0s_1 = embeddings_1[(subclasses_1 == 2) & (misclassified_1 == 0)]
    red0s_1 = embeddings_1[(subclasses_1 == 3) & (misclassified_1 == 1)]
    blue0s_16 = embeddings_16[(subclasses_16 == 2) & (misclassified_16 == 0)]
    red0s_16 = embeddings_16[(subclasses_16 == 3) & (misclassified_16 == 1)]

    # we expect similarity to increase
    print('At epoch 1, how similar are blue1s from red1s?', compare_embeddings(blue1s_1, red1s_1))
    print('At epoch 16, how similar are blue1s from red1s?', compare_embeddings(blue1s_16, red1s_16))

    # we expect similarity to decrease
    print('At epoch 1, how similar are blue1s to blue0s?', compare_embeddings(blue1s_1, blue0s_1))
    print('At epoch 16, how similar are blue1s to blue0s?', compare_embeddings(blue1s_16, blue0s_16)) 

    print('How similar are blue1s from epoch 1 to blue1s from epoch 16?', compare_embeddings(blue1s_1, blue1s_16))

    # Now, let's subtract epoch 1 embeddings from epoch 16 embeddings.
    diff_blue0s = blue0s_16 - np.mean(blue0s_1, axis=0)
    diff_blue1s = blue1s_16 - np.mean(blue1s_1, axis=0)
    diff_red0s = red0s_16 - np.mean(red0s_1, axis=0)
    diff_red1s = red1s_16 - np.mean(red1s_1, axis=0)

    print('After masking out epoch 1, how similar are blue0s and blue1s?', compare_embeddings(diff_blue0s, diff_blue1s))
    print('After masking out epoch 1, how similar are blue0s and red0s?', compare_embeddings(diff_blue0s, diff_red0s))

    print('After masking out epoch 1, how similar are red0s and red1s?', compare_embeddings(diff_red0s, diff_red1s))
    print('After masking out epoch 1, how  similar are red1s and blue1s?', compare_embeddings(diff_red1s, diff_blue1s))


def group_accuracy_score(y, y_pred, subclasses):
    return {'Blue 1s': accuracy_score(y[subclasses == 0], y_pred[subclasses == 0]),
    'Red 1s': accuracy_score(y[subclasses == 1], y_pred[subclasses == 1]),
    'Blue 0s': accuracy_score(y[subclasses == 2], y_pred[subclasses == 2]),
    'Red 0s': accuracy_score(y[subclasses == 3], y_pred[subclasses == 3]),
    'Overall Accuracy': accuracy_score(y, y_pred)}


def remap_subclasses(subclasses, test_subclasses):
    subclasses[subclasses == 0] = 10
    subclasses[subclasses == 1] = 12
    subclasses[subclasses == 2] = 11
    subclasses[subclasses == 3] = 13
    subclasses[subclasses == 10] = 0
    subclasses[subclasses == 11] = 1
    subclasses[subclasses == 12] = 2
    subclasses[subclasses == 13] = 3

    test_subclasses[test_subclasses == 0] = 10
    test_subclasses[test_subclasses == 1] = 12
    test_subclasses[test_subclasses == 2] = 11
    test_subclasses[test_subclasses == 3] = 13
    test_subclasses[test_subclasses == 10] = 0
    test_subclasses[test_subclasses == 11] = 1
    test_subclasses[test_subclasses == 12] = 2
    test_subclasses[test_subclasses == 13] = 3
    return subclasses, test_subclasses


'''
How good are the original embeddings at separating colors? {'Blue 1s': 0.9972508591065292, 'Red 1s': 1.0, 'Blue 0s': 1.0, 'Red 0s': 0.8935888962326504}
How good are the original embeddings at separating numbers? {'Blue 1s': 0.8233676975945017, 'Red 1s': 0.9056356487549148, 'Blue 0s': 0.8738379814077025, 'Red 0s': 0.836748182419035}
Color Active Neurons: [ 2  7 10 11 12 16 21 23 24 25 27 28 35 38 39 40 43 45 60 61 64 70 72 76 83]
Number Active Neurons: [ 7 15 16 22 23 24 25 28 35 37 38 41 45 48 56 60 61 62 65 67 74 81]
Common Active Neurons: [ 7 16 23 24 25 28 35 38 45 60 61]
Cosine Similarity of Hyperplanes: [[0.0536327]]
Euclidean Distance between Hyperplanes: 12.622898269873687
'''
def identify_subnetworks():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_test_16_epoch')

    # logits, embeddings, subclasses, labels, _, _ = unpack_data('CelebA_meta_train_12_epoch')
    # test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('CelebA_meta_test_12_epoch')
    # subclasses[subclasses == 0] = 10
    # subclasses[subclasses == 1] = 12
    # subclasses[subclasses == 2] = 11
    # subclasses[subclasses == 3] = 13
    # subclasses[subclasses == 10] = 0
    # subclasses[subclasses == 11] = 1
    # subclasses[subclasses == 12] = 2
    # subclasses[subclasses == 13] = 3

    # test_subclasses[test_subclasses == 0] = 10
    # test_subclasses[test_subclasses == 1] = 12
    # test_subclasses[test_subclasses == 2] = 11
    # test_subclasses[test_subclasses == 3] = 13
    # test_subclasses[test_subclasses == 10] = 0
    # test_subclasses[test_subclasses == 11] = 1
    # test_subclasses[test_subclasses == 12] = 2
    # test_subclasses[test_subclasses == 13] = 3

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses):
        d = embeddings.shape[1]
        model = LinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train(model, 10, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))


    def fancy_stuff(embeddings, test_embeddings):

        embeddings = scale_G(embeddings)
        test_embeddings = scale_G(test_embeddings)

        blue1s = embeddings[(subclasses == 0) & (misclassified == 1)]
        red1s = embeddings[(subclasses == 1) & (misclassified == 0)]
        blue0s = embeddings[(subclasses == 2) & (misclassified == 0)]
        red0s = embeddings[(subclasses == 3) & (misclassified == 1)]

        N = 200
        X_train = np.concatenate((blue1s[:N], red1s[:N], blue0s[:N], red0s[:N]), axis=0)
        X_test = scale_G(test_embeddings)
        y_train_color = np.array([1] * N + [0] * N + [1] * N + [0] * N)
        y_train_number = np.array([1] * N * 2 + [0] * N * 2)
        y_test_color = np.isin(test_subclasses, [0, 2])
        y_test_number = np.isin(test_subclasses, [0, 1])

        color_model = LogisticRegression()
        color_model.fit(X_train, y_train_color)
        y_pred_color = color_model.predict(X_test)
        accuracy_color = group_accuracy_score(y_test_color, y_pred_color, test_subclasses)
        print('How good are the original embeddings at separating colors?', accuracy_color)

        number_model = LogisticRegression()
        number_model.fit(X_train, y_train_number)
        y_pred_number = number_model.predict(X_test)
        accuracy_number = group_accuracy_score(y_test_number, y_pred_number, test_subclasses)
        print('How good are the original embeddings at separating numbers?', accuracy_number)

        color_weights = color_model.coef_.flatten()
        color_bias = color_model.intercept_[0]
        number_weights = number_model.coef_.flatten()

        # color_active_neurons = np.sort(np.where(np.abs(color_weights) > 1)[0])
        # number_active_neurons = np.sort(np.where(np.abs(number_weights) > 1)[0])

        # print('Color Active Neurons:', color_active_neurons)
        # print('Number Active Neurons:', number_active_neurons)
        # print('Common Active Neurons:', np.intersect1d(color_active_neurons, number_active_neurons))

        # print('Cosine Similarity of Hyperplanes:', cosine_similarity(color_weights.reshape(1, -1), number_weights.reshape(1, -1)))
        # print('Euclidean Distance between Hyperplanes:', np.linalg.norm(color_weights - number_weights))

        def reflect_over_hyperplane(embeddings, weights, bias):
            # distances = np.dot(embeddings, weights) + bias
            # wrong_side_mask = distances < 0
            # embeddings_reflected = embeddings.copy()
            # embeddings_reflected[wrong_side_mask] -= 2 * (distances[wrong_side_mask][:, np.newaxis] * weights) / np.linalg.norm(weights) ** 2

            u = weights.reshape(-1, 1) / np.linalg.norm(weights)
            P = np.eye(84) - u @ u.T
            embeddings_reflected = embeddings @ P
            return scale_G(embeddings_reflected)
        
        reflected_X_train = reflect_over_hyperplane(X_train, color_weights, color_bias)
        reflected_X_test = reflect_over_hyperplane(X_test, color_weights, color_bias)

        color_model = LogisticRegression()
        color_model.fit(reflected_X_train, y_train_color)
        y_pred_color = color_model.predict(reflected_X_test)
        accuracy_color = group_accuracy_score(y_test_color, y_pred_color, test_subclasses)
        print('How good are the reflected embeddings at separating colors?', accuracy_color)

        number_model = LogisticRegression()
        number_model.fit(reflected_X_train, y_train_number)
        y_pred_number = number_model.predict(reflected_X_test)
        accuracy_number = group_accuracy_score(y_test_number, y_pred_number, test_subclasses)
        print('How good are the reflected embeddings at separating numbers?', accuracy_number)

        # print('Before reflection - How similar are blue1s and red1s?', compare_embeddings(X_test[test_subclasses == 0], X_test[test_subclasses == 1]))
        # print('After reflection - How similar are blue1s and red1s?', compare_embeddings(reflected_X_test[test_subclasses == 0], reflected_X_test[test_subclasses == 1]))

        reflected_embeddings = torch.tensor(reflect_over_hyperplane(embeddings, color_weights, color_bias), dtype=torch.float32)
        reflected_test_embeddings = torch.tensor(reflect_over_hyperplane(test_embeddings, color_weights, color_bias), dtype=torch.float32)
        print('Retraining (without reweighting) the last layer before reflecting embeddings over hyperplane....')
        retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)
        print('Retraining (without reweighting) the last layer after reflecting embeddings over hyperplane...')
        retrain_last_layer(reflected_embeddings, reflected_test_embeddings, labels, test_labels, test_logits, test_subclasses)

        return reflected_embeddings.numpy(), reflected_test_embeddings.numpy()

    # print('Retraining (without reweighting) the last layer before reflecting embeddings over hyperplane....')
    # retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)

    for _ in range(1):
        embeddings, test_embeddings = fancy_stuff(embeddings, test_embeddings)

    # print('Retraining (without reweighting) the last layer after reflecting embeddings over hyperplane...')
    # retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)

def identify_subnetworks_nonlinear():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, losses = unpack_data('CelebA_meta_train_12_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, test_losses = unpack_data('CelebA_meta_test_12_epoch')

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

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses):
        d = embeddings.shape[1]
        model = LinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train(model, 5, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))


    def fancy_stuff(embeddings, test_embeddings, retrain=False):

        # embeddings = np.concatenate((embeddings, losses.reshape(-1, 1)), axis=1)
        # test_embeddings = np.concatenate((test_embeddings, test_losses.reshape(-1, 1)), axis=1)
        embeddings = scale_G(embeddings)
        test_embeddings = scale_G(test_embeddings)

        # A = compute_covariance(embeddings)
        # v, e = get_real_eig(A)
        # P = e[:, -2:]
        # embeddings = embeddings @ (P @ P.T)
        # A = compute_covariance(test_embeddings)
        # v, e = get_real_eig(A)
        # P = e[:, -2:]
        # test_embeddings = test_embeddings @ (P @ P.T)


        female_brunettes = embeddings[(subclasses == 0)] # & (misclassified == 1)]
        female_blondes = embeddings[(subclasses == 1)] # & (misclassified == 0)]
        male_brunettes = embeddings[(subclasses == 2)]# & (misclassified == 0)]
        male_blondes = embeddings[(subclasses == 3)]# & (misclassified == 1)]

        N = 700
        X_train = np.concatenate((female_brunettes[:N], female_blondes[:N], male_brunettes[:N], male_blondes[:N]), axis=0)
        X_test = test_embeddings
        y_train_color = np.array([1] * N + [0] * N + [1] * N + [0] * N)
        y_train_gender = np.array([1] * N * 2 + [0] * N * 2)
        y_test_color = np.isin(test_subclasses, [0, 2])
        y_test_gender = np.isin(test_subclasses, [0, 1])

        color_model = LogisticRegression()
        color_model.fit(X_train, y_train_color)
        y_pred_color = color_model.predict(X_test)
        accuracy_color = group_accuracy_score(y_test_color, y_pred_color, test_subclasses)
        print('How good are the original embeddings at separating colors?', accuracy_color)

        gender_model = LogisticRegression()
        gender_model.fit(X_train, y_train_gender)
        y_pred_gender = gender_model.predict(X_test)
        accuracy_gender = group_accuracy_score(y_test_gender, y_pred_gender, test_subclasses)
        print('How good are the original embeddings at separating genders?', accuracy_gender)

        color_weights = color_model.coef_.flatten()
        color_bias = color_model.intercept_[0]
        gender_weights = gender_model.coef_.flatten()
        gender_bias = gender_model.coef_.flatten()

        # color_active_neurons = np.sort(np.where(np.abs(color_weights) > 1)[0])
        # number_active_neurons = np.sort(np.where(np.abs(number_weights) > 1)[0])

        # print('Color Active Neurons:', color_active_neurons)
        # print('Number Active Neurons:', number_active_neurons)
        # print('Common Active Neurons:', np.intersect1d(color_active_neurons, number_active_neurons))

        # print('Cosine Similarity of Hyperplanes:', cosine_similarity(color_weights.reshape(1, -1), number_weights.reshape(1, -1)))
        # print('Euclidean Distance between Hyperplanes:', np.linalg.norm(color_weights - number_weights))

        def reflect_over_hyperplane(embeddings, weights, bias):
            # distances = np.dot(embeddings, weights) + bias
            # wrong_side_mask = distances < 0
            # embeddings_reflected = embeddings.copy()
            # embeddings_reflected[wrong_side_mask] -= 2 * (distances[wrong_side_mask][:, np.newaxis] * weights) / np.linalg.norm(weights) ** 2

            d = embeddings.shape[1]
            u = weights.reshape(-1, 1) / np.linalg.norm(weights)
            P = np.eye(d) - u @ u.T
            embeddings_reflected = embeddings @ P
            return scale_G(embeddings_reflected)
        
        # gender_weights -= color_weights
        
        reflected_X_train = reflect_over_hyperplane(X_train, gender_weights, gender_bias)
        reflected_X_test = reflect_over_hyperplane(X_test, gender_weights, gender_bias)

        # y_train_group = np.array([0] * N + [1] * N + [2] * N + [3] * N)

        # pca = PCA(n_components=2)
        # embeddings_2d = pca.fit_transform(reflected_X_train)

        # x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
        # y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
        #                     np.arange(y_min, y_max, 0.01))

        # clf = LogisticRegression()
        # clf.fit(embeddings_2d, y_train_color)
        # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)

        # plt.figure(figsize=(10, 8))
        # plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=np.linspace(Z.min(), Z.max(), num=3))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_train_group, cmap='viridis', alpha=0.7)
        # plt.colorbar(scatter, label='Group')
        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.title('PCA of 84-Dimensional Embeddings')
        # plt.savefig('8_23.png')

        # exit()

        color_model = LogisticRegression()
        color_model.fit(reflected_X_train, y_train_color)
        y_pred_color = color_model.predict(reflected_X_test)
        accuracy_color = group_accuracy_score(y_test_color, y_pred_color, test_subclasses)
        print('How good are the reflected embeddings at separating colors?', accuracy_color)

        gender_model = LogisticRegression()
        gender_model.fit(reflected_X_train, y_train_gender)
        y_pred_gender = gender_model.predict(reflected_X_test)
        accuracy_gender = group_accuracy_score(y_test_gender, y_pred_gender, test_subclasses)
        print('How good are the reflected embeddings at separating numbers?', accuracy_gender)

        print('Before reflection - How similar are male brunettes and male blondes?', compare_embeddings(X_train[(y_train_color == 1) & (y_train_gender == 0)], X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('Before reflection - How similar are male blondes and male blondes?', compare_embeddings(X_train[(y_train_color == 0) & (y_train_gender == 0)], X_train[(y_train_color == 0) & (y_train_gender == 0)], True))
        print('Before reflection - How similar are female blondes and male blondes?', compare_embeddings(X_train[(y_train_color == 0) & (y_train_gender == 1)], X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('Before reflection - How similar are female brunettes and male blondes?', compare_embeddings(X_train[(y_train_color == 1) & (y_train_gender == 1)], X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('Before reflection - How similar are female brunettes and male brunettes?', compare_embeddings(X_train[(y_train_color == 1) & (y_train_gender == 1)], X_train[(y_train_color == 1) & (y_train_gender == 0)]))
        print('Before reflection - How similar are female brunettes and female blondes?', compare_embeddings(X_train[(y_train_color == 1) & (y_train_gender == 1)], X_train[(y_train_color == 0) & (y_train_gender == 1)]))

        print('After reflection - How similar are male brunettes and male blondes?', compare_embeddings(reflected_X_train[(y_train_color == 1) & (y_train_gender == 0)], reflected_X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('After reflection - How similar are male blondes and male blondes?', compare_embeddings(reflected_X_train[(y_train_color == 0) & (y_train_gender == 0)], reflected_X_train[(y_train_color == 0) & (y_train_gender == 0)], True))
        print('After reflection - How similar are female blondes and male blondes?', compare_embeddings(reflected_X_train[(y_train_color == 0) & (y_train_gender == 1)], reflected_X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('After reflection - How similar are female brunettes and male blondes?', compare_embeddings(reflected_X_train[(y_train_color == 1) & (y_train_gender == 1)], reflected_X_train[(y_train_color == 0) & (y_train_gender == 0)]))
        print('After reflection - How similar are female brunettes and male brunettes?', compare_embeddings(reflected_X_train[(y_train_color == 1) & (y_train_gender == 1)], reflected_X_train[(y_train_color == 1) & (y_train_gender == 0)]))
        print('After reflection - How similar are female brunettes and female blondes?', compare_embeddings(reflected_X_train[(y_train_color == 1) & (y_train_gender == 1)], reflected_X_train[(y_train_color == 0) & (y_train_gender == 1)]))



        reflected_embeddings = torch.tensor(reflect_over_hyperplane(embeddings, color_weights, color_bias), dtype=torch.float32)
        reflected_test_embeddings = torch.tensor(reflect_over_hyperplane(test_embeddings, color_weights, color_bias), dtype=torch.float32)
        # print('Retraining (without reweighting) the last layer before reflecting embeddings over hyperplane....')
        # retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)
        if retrain:
            print('Retraining (without reweighting) the last layer after reflecting embeddings over hyperplane...')
            retrain_last_layer(reflected_embeddings, reflected_test_embeddings, labels, test_labels, test_logits, test_subclasses)

        return reflected_embeddings.numpy(), reflected_test_embeddings.numpy()

    # print('Retraining (without reweighting) the last layer before reflecting embeddings over hyperplane....')
    # retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)

    for i in range(1):
        embeddings, test_embeddings = fancy_stuff(embeddings, test_embeddings, i == 10)
        print('============================================================================================')

    print('Retraining (without reweighting) the last layer after reflecting embeddings over hyperplane...')
    retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)
    

def label_swapping():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, losses = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_test_16_epoch')

    embeddings = scale_G(embeddings)
    test_embeddings = scale_G(test_embeddings)

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses, model=None):
        d = embeddings.shape[1]
        if model is None:
            model = LinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train(model, 10, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))
        return model

    idxs = np.where(((subclasses == 1) | (subclasses == 2)) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.475
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip = random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 0) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.005
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 3) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.03
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)


    for idx in idxs_to_flip:
        labels[idx] = 1 - labels[idx]


    retrained_model = retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)


def label_swapping_demographic_free():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, losses = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_test_16_epoch')

    embeddings = scale_G(embeddings)
    test_embeddings = scale_G(test_embeddings)

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses, model=None):
        d = embeddings.shape[1]
        if model is None:
            model = LinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train(model, 20, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))
        return model

    idxs = np.where((losses < 0.7) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.483
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip = random.sample(idxs, num_to_flip)

    for idx in idxs_to_flip:
        labels[idx] = 1 - labels[idx]

    retrained_model = retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)


def label_swapping_CelebA():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, losses = unpack_data('CelebA_meta_train_12_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('CelebA_meta_test_12_epoch')

    embeddings = scale_G(embeddings)
    test_embeddings = scale_G(test_embeddings)

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

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

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses, model=None):
        d = embeddings.shape[1]
        if model is None:
            model = LinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train(model, 10, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))
        return model

    idxs = np.where((subclasses == 0) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.43
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip = random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 2) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.475
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 1) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.005
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)

    # idxs = np.where((subclasses == 3) & (misclassified == 0))[0].tolist()
    # percent_to_flip = 0.03
    # num_to_flip = int(percent_to_flip * len(idxs))
    # idxs_to_flip += random.sample(idxs, num_to_flip)


    for idx in idxs_to_flip:
        labels[idx] = 1 - labels[idx]


    retrained_model = retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)


def dpsgd_lastlayer():
    def train_dpsgd(model, epochs, trainloader, optimizer, criterion):
        epoch_loss = ValueError()
        accuracy = ValueError()
        for epoch in range(epochs):
            epoch_loss = 0
            accuracy = 0
            for X_batch, y_batch in trainloader:
                optimizer.zero_grad()
                for x, y in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_batch, y_batch)):
                    logits = model(x)
                    accurate = int(torch.argmax(logits) == y)
                    accuracy += accurate
                    loss = criterion(logits, y)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.per_sample_gradient_clip()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(trainloader)}')

    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_test_16_epoch')

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    d = embeddings.shape[1]
    model = LinearModel(d, 2)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    batch_size = 32
    optimizer = DPSGD(model.parameters(), lr=0.001, noise_scale=2, group_size=batch_size, grad_norm_bound=0.00001)

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    train_dataset = TensorDataset(embeddings, labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_dpsgd(model, 5, train_loader, optimizer, criterion)
    test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
    evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))


def label_swapping_nonlinear():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, losses = unpack_data('ColoredMNIST_HARD_meta_train_16_epoch')
    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('ColoredMNIST_HARD_meta_test_16_epoch')

    embeddings = scale_G(embeddings)
    test_embeddings = scale_G(test_embeddings)

    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    class NonLinearModel(nn.Module):
        def __init__(self, d, num_classes):
            super(NonLinearModel, self).__init__()
            self.h = int(84 * 2)
            self.fc = nn.Linear(d, self.h)
            self.relu = nn.ReLU()
            self.out = nn.Linear(self.h, num_classes)

            self.post_swap_embeddings = []


        def forward(self, x, cache=False):
            x = self.fc(x)
            x = self.relu(x)
            if cache:
                self.post_swap_embeddings += [x.detach().numpy()]
            x = self.out(x)
            return x
        

    def train_cache(model, num_epochs, trainloader, optimizer, criterion):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            cache = epoch == (num_epochs - 1)
            model.post_swap_embeddings = []
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = model.forward(inputs, cache)
                loss = criterion(outputs, labels)
                running_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
            
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    def retrain_last_layer(embeddings, test_embeddings, labels, test_labels, test_logits, test_subclasses, model=None):
        d = embeddings.shape[1]
        if model is None:
            model = NonLinearModel(d, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        batch_size = 32
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = TensorDataset(embeddings, labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        train_cache(model, 10, train_loader, optimizer, criterion)
        test_predictions = torch.tensor(np.argmax(test_logits, axis=1), dtype=torch.long)
        evaluate(model, test_embeddings, torch.tensor(test_labels, dtype=torch.long), test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))
        return model

    idxs = np.where(((subclasses == 1) | (subclasses == 2)) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.480
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip = random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 0) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.01
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)

    idxs = np.where((subclasses == 3) & (misclassified == 0))[0].tolist()
    percent_to_flip = 0.15
    num_to_flip = int(percent_to_flip * len(idxs))
    idxs_to_flip += random.sample(idxs, num_to_flip)


    for idx in idxs_to_flip:
        labels[idx] = 1 - labels[idx]


    # retrained_model = retrain_last_layer(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(test_embeddings, dtype=torch.float32), labels, test_labels, test_logits, test_subclasses)
    # embeddings = scale_G(np.concatenate([batch for batch in retrained_model.post_swap_embeddings], axis=0))
    
    blue1s = embeddings[(subclasses == 0)]# & (misclassified == 1)]
    red1s = embeddings[(subclasses == 1)]# & (misclassified == 0)]
    blue0s = embeddings[(subclasses == 2)]# & (misclassified == 0)]
    red0s = embeddings[(subclasses == 3)]# & (misclassified == 1)]

    N = min(len(blue1s), len(red1s), len(blue0s), len(red0s))
    X_train = np.concatenate((blue1s[:N], red1s[:N], blue0s[:N], red0s[:N]), axis=0)
    y_train_group = np.array([0] * N + [1] * N + [2] * N + [3] * N)

    train_idxs = np.concatenate([np.where(subclasses == 0)[0][:N], np.where(subclasses == 1)[0][:N], np.where(subclasses == 2)[0][:N], np.where(subclasses == 3)[0][:N]])

    border_points = np.zeros_like(y_train_group)
    for i in range(len(y_train_group)):
        if train_idxs[i] in idxs_to_flip:
            border_points[i] = 1

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(X_train)
    print(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_train_group, edgecolors=['black' if b else 'none' for b in border_points], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Group')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of 168-Dimensional Embeddings')
    # plt.savefig('9_12_post_flip.png')




# compare_embeddings_across_epochs()
# identify_subnetworks()
# identify_subnetworks_nonlinear()
# label_swapping()
# label_swapping_demographic_free()
# label_swapping_CelebA()
dpsgd_lastlayer()
# label_swapping_nonlinear()




'''
Can't do saliency map stuff with this dataset.


do what i did but project out stuff
'''