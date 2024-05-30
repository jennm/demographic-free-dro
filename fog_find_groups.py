import numpy as np

from collections import defaultdict
from get_embeddings import create_dataloader
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from random.random import random

def fog_algorithm(pen_emb, val_pen_emb, val_last_emb, val_labels, val_loss, idx, val_idx):
    groups = defaultdict(lambda: [0])
    # for i in range(10):
    X = val_pen_emb
    # y = val_subclasses == i
    y_misclassified = (np.argmax(val_last_emb, axis=1) != val_labels)
    # y = (np.argmax(last_emb, axis=1) != labels)
    print('sanity check, training acc:', (np.argmax(val_last_emb, axis=1) == val_labels).sum() / len(val_labels))
    # X_train, _, y_train, _ = train_test_split(X, y, test_size=2/3, random_state=42)
    points_found = list()
    y_misclassified = (np.argmax(val_last_emb, axis=1) != val_labels)
    misclassified_indices = list()
    misclassified_loss = dict()
    idx_to_loss = dict()
    for i in range(len(y_misclassified)):
        if y_misclassified[i]:
            misclassified_indices.append(i)
            idx_to_loss[i] = val_loss[i]
            if val_loss[i] in misclassified_loss:
                misclassified_loss[val_loss[i]].append(i)
            else:
                misclassified_loss[val_loss[i]] = [i]
    misclassified_indices = np.array(misclassified_indices)
    y_misclassified = np.array(y_misclassified)
    X_train = X.copy()
    Y_train = np.array([0 for i in range(len(X_train))])
    sorted_misclassified_losses = list(misclassified_loss.keys())
    sorted_misclassified_losses.sort()
    # removed_points = list()
    count = 0
    while count < 5:#10:
        # for j in range(10):
        # misclassified_indices = list()
        # for i in range(len(y_misclassified)):
        #   if y_misclassified[i]: misclassified_indices.append(i)
        # y_misclassified = np.array(y_misclassified)

        print('FINDING NEW GROUP')
        # y_misclassified = (np.argmax(val_last_emb, axis=1) != val_labels)
        # misclassified_indices = list()
        # for i in range(len(y_misclassified)):
        #   if y_misclassified[i]: misclassified_indices.append(i)
        # points_found += misclassified_indices
        print(len(y_misclassified))
        if len(y_misclassified) == 0: break


        # misclassified_point = random.choice(misclassified_loss[sorted_misclassified_losses[-1]])
        print(misclassified_point)
        misclassified_point = random.choice(misclassified_indices) # old way of selecting misclassified point


        # find points close by
        distances = list()
        dist_dict = dict()
        for i in range(len(X_train)):
            if i == misclassified_point: continue
            # print(X_train[misclassified_point], X_train[i])
            dist = distance.euclidean(X_train[misclassified_point], X_train[i])
            # torch.pdist(X[misclassified_point], X[i])
            distances.append(dist)
            if dist in dist_dict:
                dist_dict[dist].append(i)
            else:
                dist_dict[dist] = [i]

        distances.sort()
        top_five = round(len(distances) * .05 + .5)
        # print("top five", top_five)
        distance_set = set(distances[0:top_five])
        # removed_points = list()
        y = [0 for i in range(len(X_train))]
        for dist in distance_set:
            for i in dist_dict[dist]:
                y[i] = 1
                # removed_points.append(i)
        y = np.array(y)#, dtype=torch.float) #idk if this is the right type
        y_train = y.copy()


        # X_train, _, y_train, _ = train_test_split(X, y, test_size=2/3, random_state=42)
        groups, remove_misclassified, remove_all = fog_algorithm_subgroup_finder(X=X, y=y, y_misclassified=y_misclassified, X_train=X_train, y_train=y_train, threshold=0.1, groups=groups, pen_emb=pen_emb, val_pen_emb=val_pen_emb, idx=idx, val_idx=val_idx)
        if len(remove_all) == 0: break
        # print(len(y_train), len(remove_misclassified))
        # if remove_misclassified.sum() < 3:
        #   continue
        count += 1
        X_train = X_train[~remove_misclassified]
        y_train = y_train[~remove_misclassified]
        y_misclassified = y_misclassified[~remove_misclassified]

        # for i in range(len(remove_misclassified)):
        #   if remove_misclassified[i] and i in idx_to_loss:
        #     loss = idx_to_loss.pop(i)
        #     misclassified_loss[loss].remove(i)
        #     if len(misclassified_loss[loss]) == 0:
        #       misclassified_loss.pop(loss)
        #       sorted_misclassified_losses.remove(loss)

        # print(misclassified_indices)
        # print(remove_misclassified)
        to_remove = list()
        new_misclassified_indices = list()
        cur_idx = 0
        for loc in range(len(misclassified_indices)):
            i = misclassified_indices[loc]
            # if i == 5011:
            #   print('i', remove_misclassified[i], idx_to_loss[i])
            if remove_misclassified[i] and i in idx_to_loss:
                loss = idx_to_loss.pop(i)
                misclassified_loss[loss].remove(i)
                if len(misclassified_loss[loss]) == 0:
                    misclassified_loss.pop(loss)
                    sorted_misclassified_losses.remove(loss)
            if not remove_misclassified[i]:
                if i in idx_to_loss:
                    loss = idx_to_loss[i]
                    idx_to_loss.pop(i)
                    idx_to_loss[cur_idx] = loss
                    misclassified_loss[loss].remove(i)
                    misclassified_loss[loss].append(cur_idx)
                new_misclassified_indices.append(cur_idx)
                cur_idx += 1

        misclassified_indices = np.array(new_misclassified_indices)
        print(len(misclassified_indices))
        if len(misclassified_indices) == 0 or y_misclassified.sum() == 0: break
        print(y_train.sum())#, y_train)
        if y_train.sum() == 0 or y_train.sum() == 1: break

        print(groups)
        
def fog_algorithm_subgroup_finder(X, y, y_misclassified, X_train, y_train, threshold, groups, pen_emb, val_pen_emb, idx, val_idx):
    group_num = 1 if len(groups) == 0 else len(groups[0])

    model, y_pred = train_classifier(X, X_train, y_train)
    # evaluate_classifier(y, y_pred)

    for i in range(3):
      y_train = update_misclassified(model, X_train, y_train, threshold)
      model, y_pred = train_classifier(X, X_train, y_train)
      # evaluate_classifier(y, y_pred)


    y_pred = model.predict(X_train)

    # ERM_misclassified_in_positive_class = (y_pred == 1) & (y_train == 1)
    misclassified_remove = ((y == 1) | (y_misclassified == 1)) & (y_pred == 1)
    # print('size diff', ((y == 1) & (y_pred == 1)).sum(), ((y_misclassified == 1) & (y_pred == 1)).sum(), (((y_misclassified == 1) | (y == 1)) & (y_pred == 1)).sum())
    positive_class = (y_pred == 1)

    X_train = pen_emb
    y_pred = model.predict(X_train)

    for pt in idx:
      if y_pred[pt]:
          groups[pt].append(group_num) # add group num
      else:
        groups[pt].append(-1) # no group


    X_val = val_pen_emb
    val_y_pred = model.predict(X_val)
    for pt in val_idx:
      if val_y_pred[pt - len(X_train)]:# and val_y_pred[pt - len(X)]: # if misclassified and in positive class
        groups[pt].append(group_num) # put in group
      else: groups[pt].append(-1) # else don't
      # groups[pt].append(is_misclassified * group_num * val_y_pred[pt - len(X)] - 1 * (not val_y_pred[pt - len(X)]))

    return groups, misclassified_remove, positive_class

def train_classifier(X, X_train, y_train):
    class_weight = 'balanced'
    model = LogisticRegression(class_weight=class_weight, penalty='l1', solver='liblinear')

    model.fit(X_train, y_train)
    y_pred = model.predict(X)

    return model, y_pred

def update_misclassified(model, X_train, y_train, threshold): # does the classifier get me wrong and am i within some distance of the threshold
  y_train_pred = model.predict(X_train)
  misclassified = (y_train_pred != y_train)
  distances = np.abs(model.decision_function(X_train))
  misclassified_below_threshold = (misclassified) & (distances < threshold)
  y_train[misclassified_below_threshold] = 1 - y_train[misclassified_below_threshold]

  return y_train

def evaluate_classifier(y, y_pred, val_subclasses):
    print('% misclassified group 0 that ends up in positive class:', ((y_pred == 1) & (val_subclasses == 0) & (y == 1)).sum() / ((val_subclasses == 0) & (y == 1)).sum())
    print('% misclassified group 3 that ends up in positive class:', ((y_pred == 1) & (val_subclasses == 3) & (y == 1)).sum() / ((val_subclasses == 3) & (y == 1)).sum())
    print('% misclassified rare points that end up in positive class', ((y_pred == 1) & ((val_subclasses == 0) | (val_subclasses == 3)) & (y == 1)).sum() / (((val_subclasses == 0) | (val_subclasses == 3)) & (y == 1)).sum())

    # Evaluate the model accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy at separating misclassified points based on full embedding: {accuracy:.2f}')

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate true positive rate
    tpr = tp / (tp + fn)

    # Display true positive rate
    print(f'True Positive Rate (TPR): {tpr:.2f}')
    print(tp, fp, tn, fn)