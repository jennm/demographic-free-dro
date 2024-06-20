import numpy as np
from collections import Counter
from scipy.spatial import distance

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA, FastICA

from collections import Counter

def unpack_data(part):
    train_npz = np.load(f'cmnist_meta_{part}.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    losses = train_npz['loss']
    
    return predictions, embeddings, subclasses, labels, data_idx, losses


def loss_report(subclasses, losses, misclassified):
    print()
    print('================================LOSS REPORT START================================')
    print(f'Min Loss: {np.min(losses)}, Max Loss: {np.max(losses)}')
    print('Subclass of Point with Highest Loss:', subclasses[np.argmax(losses)])

    class_losses = {}
    class_counts = {}
    for c, l in zip(subclasses, losses):
        if c in class_losses: 
            class_losses[c] += l
            class_counts[c] += 1
        else: 
            class_losses[c] = l
            class_counts[c] = 1

    class_avg_losses = {c: class_losses[c] / class_counts[c] for c in class_losses}
    sorted_classes = sorted(class_avg_losses.items(), key = lambda x: x[1], reverse=True)
    
    class_loss_ranges = {c: (np.min(np.append(losses[(subclasses == c) & (misclassified)], 1000)), np.max(np.append(losses[(subclasses == c) & (misclassified)], -1000))) for c in range(25)}

    print("Classes with highest average losses:")
    for i, (c, avg_loss) in enumerate(sorted_classes, 1):
        print(f'{i}. Class {c}: Average Loss = {avg_loss:.2f}; Misclassified Loss Range: {class_loss_ranges[c]}')

    print('================================LOSS REPORT END================================')
    print()

def neighborhood_exploration():
    # load data
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')
    misclassified = np.argmax(predictions, axis=1) != labels

    print('# Misclassified Points in Full Dataset:', misclassified.sum())

    row_means = np.mean(embeddings, axis=1, keepdims=True)
    row_stds = np.std(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings - row_means) / row_stds

    arr_mean = np.mean(losses)
    arr_std = np.std(losses)
    losses = (losses - arr_mean) / arr_std

    # pca = PCA(n_components=42)
    # embeddings = pca.fit_transform(embeddings)
    # print('Reduced Embedding Shape:', embeddings.shape)

    embeddings = np.hstack((embeddings, losses.reshape(-1, 1)))

    label_majority_group = 18
    label_minority_groups = [15, 16, 17, 19]

    # calculate which subclasses have the highest average loss
    # loss_report(subclasses, losses, misclassified)    

    # pick a random label minority group
    threshold = 0.80
    chosen_label = 3 # np.random.choice(label_minority_groups)
    print('Chosen Label:', chosen_label)
    print()

    misclassified_chosen_label_mask = (misclassified) & (subclasses == chosen_label) # i am misclassified and of subgroup X

    # pick a random misclassified point with the chosen label
    seed_idx = np.random.choice(data_idx[misclassified_chosen_label_mask])

    # get the 50 nearest misclassified points in embedding space
    misclassified_embeddings = embeddings[misclassified]
    misclassified_subclasses = subclasses[misclassified]
    misclassified_data_idx = data_idx[misclassified]

    misclassified_seed_idx = np.where(data_idx[misclassified] == seed_idx)[0][0] # gets the index relative to the misclassified array
    
    assert misclassified_data_idx[misclassified_seed_idx] == seed_idx, "you're not looking at the same seed anymore"

    print()

    distance_metric = 'cosine'

    # euclidean distance = L2 norm
    misclassified_distances = distance.cdist([misclassified_embeddings[misclassified_seed_idx]], misclassified_embeddings, distance_metric).flatten()

    misclassified_train_set_size = 1001 # 1001
    nearest_misclassified_indices = np.argsort(misclassified_distances)[1:misclassified_train_set_size] # gets the indices relative to the misclassified array
    nearest_misclassified_embeddings = misclassified_embeddings[nearest_misclassified_indices]
    nearest_misclassified_subclasses = misclassified_subclasses[nearest_misclassified_indices]

    #############################################################################################
    correct_embeddings = embeddings[~misclassified] 
    correct_subclasses = subclasses[~misclassified]
    correct_distances = distance.cdist([misclassified_embeddings[misclassified_seed_idx]], correct_embeddings, distance_metric).flatten()
    correct_train_set_size = 5001 # 5001
    nearest_correct_indices = np.argsort(correct_distances)[1:correct_train_set_size] # gets the indices relative to the ~misclassified array
    nearest_correct_embeddings = correct_embeddings[nearest_correct_indices]
    nearest_correct_subclasses = correct_subclasses[nearest_correct_indices]
    #############################################################################################
    
    repeat = 1

    # train an LR to separate misclassified points of the chosen label from the rest of the misclassified neighborhood
    # X_train = nearest_misclassified_embeddings
    X_train = np.concatenate((np.repeat(nearest_misclassified_embeddings, repeat, axis=0), nearest_correct_embeddings)) # repeats misclassified embeddings X times and concatenates with correct embeddings

    # y_train = (nearest_misclassified_subclasses == chosen_label).astype(int) 
    y_train = (np.concatenate((np.repeat(nearest_misclassified_subclasses, repeat, axis=0), nearest_correct_subclasses)) == chosen_label).astype(int) # find where concatenation is chosen subclass

    # get "names" of all points used in training set
    train_idx = np.array([np.where(data_idx == value)[0][0] for value in np.concatenate(
             (
                 np.repeat(misclassified_data_idx[nearest_misclassified_indices], repeat), 
                (data_idx[~misclassified])[nearest_correct_indices]
             )
        )])
    
    # this gives us the indices relative to data_idx for the points in our train set
    
    
    test_idx = []
    for idx in range(len(embeddings)):
        if idx not in train_idx: test_idx += [idx]

    test_idx = np.array(test_idx)
    X_test = embeddings[test_idx]
    test_subclasses = subclasses[test_idx]
    y_test = test_subclasses == chosen_label
    
    train_misclassified = misclassified[train_idx]
    test_misclassified = misclassified[test_idx]

    assert np.array_equal(train_misclassified, np.concatenate((np.ones_like(np.repeat(nearest_misclassified_indices, repeat)), np.zeros_like(nearest_correct_indices))))

    # y_train = (y_train) & train_misclassified # (np.concatenate((np.ones_like(np.repeat(nearest_misclassified_indices, repeat)), np.zeros_like(nearest_correct_indices))))
    # y_test = (y_test) & (test_misclassified)

    print('Size of Train Set:', len(y_train))
    print('# Positive Labels in Train Set:', y_train.sum())
    print()

    print('Size of Test Set:', len(X_test))
    print('# Positive Labels in Test Set:', y_test.sum())
    print('# Misclassified Points in Test Set:', test_misclassified.sum())
    print()

    # assert test_misclassified.sum() == misclassified.sum() - len(X_train), "the LR training set is a subset of the misclassified points across the full dataset"

    model = LogisticRegression(class_weight='balanced', max_iter=500)
    model.fit(X_train, y_train)

    y_probs_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_probs_train >= threshold).astype(int)

    # y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    print('==================================TRAIN LR REPORT START==================================')
    print(f"Training Accuracy: {train_accuracy}")
    print(classification_report(y_train, y_pred_train))
    # print('True Positive = i am predicted as the chosen class and i am the chosen class, regardless of misclassified or not')
    print(f"# True Positives / Positive Labels in Train Set: {((y_pred_train == y_train) & (y_train == 1)).sum()} / {y_train.sum()}")
    print(f"# True Positives / Predicted Positives in Train Set: {((y_pred_train == y_train) & (y_train == 1)).sum()} / {y_pred_train.sum()}")
    print('==================================TRAIN LR REPORT END==================================')
    print()

    # apply the LR to the full training set to see how it generalizes
    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= threshold).astype(int)
    # y_pred_test = model.predict(X_test)

    print('==================================TEST LR REPORT START==================================')
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Testing Accuracy: {test_accuracy}")

    print(classification_report(y_test, y_pred_test))

    print(f"# True Positives / Positive Labels in Test Set: {((y_pred_test == y_test) & (y_test == 1)).sum()} / {y_test.sum()}")
    print(f"# True Positives / Predicted Positives in Test Set: {((y_pred_test == y_test) & (y_test == 1)).sum()} / {y_pred_test.sum()}")

    print('==================================TEST LR REPORT END==================================')
    print()

    predicted_positive_subclasses = test_subclasses[(y_pred_test == 1)]

    print('Predicted Positive Class Breakdown')
    subclass_counts = Counter(predicted_positive_subclasses)
    sorted_subclasses = sorted(subclass_counts.items(), key = lambda x: x[1], reverse=True)

    for i, (subclass, count) in enumerate(sorted_subclasses, 1):
        print(f"{i}. {subclass}: {count}")

    predicted_positive_subclasses = test_subclasses[(y_pred_test == 1) & (test_misclassified)]

    print('Misclassified Predicted Positive Class Breakdown')
    subclass_counts = Counter(predicted_positive_subclasses)
    sorted_subclasses = sorted(subclass_counts.items(), key = lambda x: x[1], reverse=True)

    for i, (subclass, count) in enumerate(sorted_subclasses, 1):
        print(f"{i}. {subclass}: {count}")

    ######################################################################################################################
    # save cluster file
    print()
    print('==================================STATS ON CMNIST TRAIN=================================')
    # run this on mnist train
    predictions, embeddings, subclasses, labels, data_idx, losses = unpack_data('train')
    misclassified = np.argmax(predictions, axis=1) != labels

    row_means = np.mean(embeddings, axis=1, keepdims=True)
    row_stds = np.std(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings - row_means) / row_stds

    arr_mean = np.mean(losses)
    arr_std = np.std(losses)
    losses = (losses - arr_mean) / arr_std

    # pca = PCA(n_components=42)
    # embeddings = pca.fit_transform(embeddings)
    # print('Reduced Embedding Shape:', embeddings.shape)

    embeddings = np.hstack((embeddings, losses.reshape(-1, 1)))

    X_test = embeddings
    test_subclasses = subclasses
    y_test = test_subclasses == chosen_label

    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= threshold).astype(int)
    # y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(test_accuracy)
    print(classification_report(y_test, y_pred_test))

    predicted_positive_subclasses = test_subclasses[(y_pred_test == 1)]

    print('Predicted Positive Class Breakdown')
    subclass_counts = Counter(predicted_positive_subclasses)
    sorted_subclasses = sorted(subclass_counts.items(), key = lambda x: x[1], reverse=True)

    for i, (subclass, count) in enumerate(sorted_subclasses, 1):
        print(f"{i}. {subclass}: {count}")


    predicted_positive_subclasses = test_subclasses[(y_pred_test == 1) & (misclassified)]

    print('Misclassified Predicted Positive Class Breakdown')
    subclass_counts = Counter(predicted_positive_subclasses)
    sorted_subclasses = sorted(subclass_counts.items(), key = lambda x: x[1], reverse=True)

    for i, (subclass, count) in enumerate(sorted_subclasses, 1):
        print(f"{i}. {subclass}: {count}")

    
    # get data_idx for each point in predicted class
    group_members = data_idx[y_pred_test] & misclassified

    # write to file
    np.savez(f'group_members_{chosen_label}.npz', group=group_members)

    # np.savez(f'group_members_misclassified.npz', group=misclassified)

    # do for 15, 16, 17, 19