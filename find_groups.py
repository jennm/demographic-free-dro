import torch
from classifier import LogisticRegressionModel
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors._classification import KNeighborsClassifier

from get_embeddings import get_emb_loader



'''
NOTE: The index that the DataLoader uses when getting a datapoint is the index relative to a fixed, static underlying data store.
Shuffling just changes the order in which indices are accessed. Split just changes the order in which indices are split.
Subset takes in a subset of indices from the full dataset and indexes relative to the full dataset. It does not do any reindexing.
'''

def reshape_batch_embs(device, batch):
    layers = batch.keys() # batch is really a dictionary of layer embeddings, get keys for all layers stored
    batch_multi_emb = [] # store flattened embeddings from each layer
    for layer in layers: # for each layer
        if layer == str(0): continue
        layer_emb = torch.tensor(batch[layer], device=device) # get the layer embeddings
        batch_multi_emb.append(layer_emb.view(layer_emb.size(0), -1)) # append the flattened embeddings i.e. (32 x 10 x 10 => 32 x 100)
    return torch.cat(batch_multi_emb, dim=1) # concatenate the flattened embeddings i.e. (32 x 10 and 32 x 20 => 32 x 30)

def train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs):
    print('TRAINING LR CLASSIFIER')

    model.train() # model already explicitly moved to device
    for epoch in range(num_epochs): # for each epoch
        print(f'EPOCH {epoch} / {num_epochs}')

        total_loss = 0
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(device, batch['embeddings']) # reshape to combine embeddings from multiple layers
            LR_targets = torch.tensor(batch['LR_targets'], device=device) # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            
            # w1 = LR_targets.shape[0] / (LR_targets == 0).sum() if (LR_targets == 0).sum() > 0 else 0
            # w2 = LR_targets.shape[0] / (LR_targets == 1).sum() if (LR_targets == 1).sum() > 0 else 0
            # criterion.weight = torch.tensor([w1, w2], device=device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(batch_multi_emb)
            loss = criterion(outputs, LR_targets)
            loss = loss.mean()
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()

        print(f'Total Loss for Epoch {epoch}: {total_loss}')

def update_misclassified(device, train_data, train_emb_loader, model, threshold):
    print('UPDATING MISCLASSIFIED')

    model.eval()
    with torch.no_grad():

        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(device, batch['embeddings']) # reshape to combine embeddings from multiple layers

            LR_targets = torch.tensor(batch['LR_targets'], device=device) # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            outputs = model(batch_multi_emb)
            predicted = torch.argmax(outputs, dim=1)
            misclassified = predicted != LR_targets # 1 if misclassified and 0 if correct

            distances = torch.abs(torch.max(outputs, dim=1)[0] - 0.5) # see how far each output is from the decision boundary

            misclassified_distances = distances[misclassified]
            sorted_distances, _ = torch.sort(misclassified_distances)
            percentile_index = int(0.9 * len(sorted_distances))
            distance_index = min(percentile_index, len(sorted_distances) - 1)  # Ensure index does not exceed array length
            threshold = sorted_distances[distance_index].item()

            data_idx = torch.tensor(batch['idx'], device=device)
            # indices_to_update = torch.nonzero((distances < threshold) & (misclassified), as_tuple=False).squeeze() # get the indices for the distances that are within some threshold of the decision boundary
            indices_to_update = torch.nonzero((misclassified), as_tuple=False).squeeze() # get the indices for the distances that are within some threshold of the decision boundary
            
            if indices_to_update.size() == 0: return

            data_indices_to_update = data_idx[indices_to_update].cpu().numpy()
            train_data.update_LR_y(data_indices_to_update, (1 - LR_targets[indices_to_update]).cpu().numpy())

def eval_classifier(device, model, val_emb_loader):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    points_to_ignore = np.array([])
    with torch.no_grad():
        correct = 0
        total = 0

        for batch in val_emb_loader:
            batch_multi_emb = reshape_batch_embs(device, batch['embeddings']) # reshape to combine embeddings from multiple layers
            erm_predicted_labels = torch.tensor(np.argmax(batch['embeddings'][str(0)], axis=1)).to(device)
            true_labels = torch.tensor(batch['actual_targets']).to(device)

            outputs = model(batch_multi_emb)
            LR_predicted = torch.argmax(outputs, dim=1)

            all_ones = torch.ones(true_labels.size(0), device=device)
            all_zeros = torch.zeros(true_labels.size(0), device=device)

            tp += ((erm_predicted_labels != true_labels) & (LR_predicted == all_ones)).sum().item()  # how many ERM misclassified points does the classifier put in group 1
            fp += ((erm_predicted_labels == true_labels) & (LR_predicted == all_ones)).sum().item()  # how many ERM correctly classified points does the classifier put in group 1
            tn += ((erm_predicted_labels == true_labels) & (LR_predicted == all_zeros)).sum().item() # how many ERM misclassified points does the classifier put in group 0
            fn += ((erm_predicted_labels != true_labels) & (LR_predicted == all_zeros)).sum().item() # how many ERM correctly classified points does the classifier put in group 0

            total += LR_predicted.size(0)
            correct = tp + tn

            data_idx = batch['idx']
            points_to_ignore = np.concatenate((points_to_ignore, data_idx[LR_predicted.long().cpu().numpy()]))

        accuracy = correct / total
        print('TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn)
        print(f'Val Accuracy: {accuracy:.4f}')
        ppv = tp/(max(1, tp+fp))
        print(f'TPR: {tp/(max(1, tp+fn))}\tFPR: {fp/(max(1, tn+fp))}\tTNR: {tn/(max(1, tn+fp))}\tFNR: {fn/(max(1, tp+fn))}\tPPV: {ppv}\t1 - PPV: {1 - ppv}')

        return points_to_ignore

def _find_groups(data, num_epochs, k, groups):

    group_num = 1 if (len(groups) == 0) else len(groups[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pos_group_count = 0

    train_subset_emb_loader, uni_train_subset_emb_loader, uni_train_emb_loader = data['train_subset_emb_loader'], data['uni_train_subset_emb_loader'], data['uni_train_emb_loader']

    ex_batch = next(iter(train_subset_emb_loader))['embeddings'] # example batch of embeddings (this is a dictionary)
    
    ex_batch_multi_emb = reshape_batch_embs(device, ex_batch) # concatenate embeddings from multiple layers together
    multi_emb_size = ex_batch_multi_emb.shape[-1] # get size of single vector in batch
 
    model = LogisticRegressionModel(multi_emb_size, 2) # binary classifier
    model.to(device) # move LR model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_classifier(device, model, train_subset_emb_loader, criterion, optimizer, num_epochs)
    points_to_ignore = eval_classifier(device, model, uni_train_subset_emb_loader)

    for i in range(k):
        print(f'SINGLE GROUP FINDING ITER {i}/{k}')
        update_misclassified(device, data['train_subset'], uni_train_subset_emb_loader, model, threshold=None)


        # model = LogisticRegressionModel(multi_emb_size, 2) # binary classifier
        # model.to(device) # move LR model to GPU
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        train_classifier(device, model, train_subset_emb_loader, criterion, optimizer, num_epochs)

        points_to_ignore = eval_classifier(device, model, uni_train_subset_emb_loader)

    model.eval()
    with torch.no_grad():
        for loader in [uni_train_emb_loader]: #, val_emb_loader]:
            subclasses = {}
            positive_class_size = 0

            for batch in loader:
                batch_multi_emb = reshape_batch_embs(device, batch['embeddings'])
                outputs = model(batch_multi_emb)
                predicted = torch.argmax(outputs, dim=1)

                data_idx = batch['idx']

                for i, idx in enumerate(data_idx):
                    pos_group_count += predicted[i].item()
                    groups[idx].append(group_num * predicted[i].item() - 1 * (not predicted[i].item()))

                erm_predicted_labels = np.argmax(batch['embeddings'][str(0)], axis=1)
                true_labels = batch['actual_targets']
                true_subclasses = batch['group']

                misclassified = erm_predicted_labels != true_labels

                for i, (group, misclassified, positive_class) in enumerate(zip(true_subclasses, misclassified, predicted)):
                    tag = 'misclassified' if misclassified else 'correct'
                    subclasses[f'{group}_{tag}'] = subclasses.get(f'{group}_{tag}', 0) + int(positive_class)

                positive_class_size += predicted.sum()

            for found_group in subclasses:
                print(f'{found_group}: {subclasses[found_group]} / {positive_class_size}')
                    
    return groups, pos_group_count, (len(groups) - pos_group_count), points_to_ignore

def find_groups(train_data, val_data, aug_indices, feature_extractor, use_classifier_groups=False, num_epochs=5, k=0, max_iter=4, min_group=100, groups=None, **loader_kwargs):
    if not groups: groups = defaultdict(lambda: [0])
    else: 
        torch.load(groups)
        groups = {i: row.tolist() for i, row in enumerate(groups)}

    pos_count, neg_count = float('inf'), float('inf')

    data = {}

    train_data.create_LR_y()
    train_subset, train_subset_emb_loader, uni_train_subset_emb_loader, uni_train_emb_loader = get_emb_loader(train_data, aug_indices, feature_extractor, train=True, ignore_points=None, version='subset + sampler', use_classifier_groups=use_classifier_groups, **loader_kwargs)
    val_subset, val_subset_emb_loader, uni_val_subset_emb_loader, uni_val_emb_loader = get_emb_loader(val_data, aug_indices, feature_extractor, train=False, ignore_points=None, version='subset + sampler', use_classifier_groups=use_classifier_groups, **loader_kwargs)


    experiment(uni_train_emb_loader, 'train')
    experiment(uni_val_emb_loader, 'val')

    return None


    data['train_subset'] = train_subset
    data['train_subset_emb_loader'] = train_subset_emb_loader # for actual training
    data['uni_train_subset_emb_loader'] = uni_train_subset_emb_loader # for updating points
    data['uni_train_emb_loader'] = uni_train_emb_loader # just so we can assign all group info
    
    run = 0
    while pos_count > min_group and neg_count > min_group and run < max_iter:
        print('FINDING ONE MORE GROUP')
        groups, pos_count, neg_count, ignore_points = _find_groups(data, num_epochs, k, groups)
        print('POS COUNT:', pos_count, 'NEG COUNT:', neg_count)
        run += 1

        train_subset, train_subset_emb_loader, uni_train_subset_emb_loader, uni_train_emb_loader = get_emb_loader(train_data, aug_indices, feature_extractor, train=True, ignore_points=ignore_points, version='full + sampler', use_classifier_groups=use_classifier_groups, **loader_kwargs)
        data['train_subset'] = train_subset
        data['train_subset_emb_loader'] = train_subset_emb_loader # for actual training
        data['uni_train_subset_emb_loader'] = uni_train_subset_emb_loader # for updating points
        data['uni_train_emb_loader'] = uni_train_emb_loader # just so we can assign all group info

    save_idx = list(groups.keys())
    save_idx.sort()
    store_groups = [torch.tensor(groups[i]) for i in save_idx]
    store_groups = torch.concatenate(store_groups)

    torch.save({'group_array': store_groups}, 'classifier_groups.pt')

    
    return groups

def experiment(data_loader, desc):
    store_last = []
    store_pen = []
    store_sub = []
    store_label = []
    store_data_idx = []

    for batch in data_loader:
        last_layer_emb = batch['embeddings'][str(0)]
        # print(last_layer_emb.shape)
        store_last.append(last_layer_emb)
        pen_layer_emb = batch['embeddings'][str(1)]
        store_pen.append(pen_layer_emb)
        true_subclass = batch['group']
        store_sub.append(true_subclass)
        true_label = batch['actual_targets']
        store_label.append(true_label)

        data_idx = batch['idx']
        store_data_idx.append(data_idx)
    
    store_last = np.concatenate(store_last)
    store_pen = np.concatenate(store_pen)
    store_sub = np.concatenate(store_sub)
    store_label = np.concatenate(store_label)
    store_data_idx = np.concatenate(store_data_idx)

    print((np.argmax(store_last, axis=1) == store_label).sum() / len(store_label))

    # reducer = umap.UMAP()
    # store_pen = reducer.fit_transform(store_pen)

    np.savez(f'cmnist_meta_{desc}.npz', last_layer=store_last, pen_layer=store_pen, subclass=store_sub, label=store_label, data_idx=store_data_idx)


##############################################
##############################################
##############################################

def visualize_classifier(train_emb_loader, layer, model, device):
    
    all_embeddings = []
    all_groups = []
    for batch in train_emb_loader:
        batch_embeddings = batch['embeddings'][str(layer)]
        all_embeddings.append(batch_embeddings)
        all_groups.append(torch.tensor(batch['group']))

    all_embeddings = torch.cat(all_embeddings)
    all_groups = torch.cat(all_groups)

    model.eval()
    with torch.no_grad():
        y_predicted = torch.argmax(model(all_embeddings), dim=1)
        y_predicted = y_predicted.cpu()

    all_embeddings = all_embeddings.cpu()
    reducer = umap.UMAP()
    X_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))

    num_groups = len(torch.unique(all_groups))
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    resolution = 100 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_2d[:,0]), np.max(X_2d[:,0])
    X2d_ymin, X2d_ymax = np.min(X_2d[:,1]), np.max(X_2d[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_2d, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Dim 1')
    plt.ylabel('UMAP Dim 2')
    plt.contourf(xx, yy, voronoiBackground)

    for i in range(num_groups):
        mask = all_groups == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, color=colors[i], label=f'Group {i}')

    plt.legend(title='Groups', loc='best')
    plt.savefig(f'visualize_classifier.png')



    
