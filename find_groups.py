import torch
from classifier import LogisticRegressionModel
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors._classification import KNeighborsClassifier



'''
NOTE: The index that the DataLoader uses when getting a datapoint is the index relative to a fixed, static underlying data store.
Shuffling just changes the order in which indices are accessed. Split just changes the order in which indices are split.
Subset takes in a subset of indices from the full dataset and indexes relative to the full dataset. It does not do any reindexing.
'''

def reshape_batch_embs(batch):
    layers = batch.keys() # batch is really a dictionary of layer embeddings, get keys for all layers stored
    batch_multi_emb = [] # store flattened embeddings from each layer
    for layer in layers: # for each layer
        if layer == str(0): continue
        layer_emb = batch[layer] # get the layer embeddings
        batch_multi_emb.append(layer_emb.view(layer_emb.size(0), -1)) # append the flattened embeddings i.e. (32 x 10 x 10 => 32 x 100)
    return torch.cat(batch_multi_emb, dim=1) # concatenate the flattened embeddings i.e. (32 x 10 and 32 x 20 => 32 x 30)

def train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs):
    print('TRAINING LR CLASSIFIER')
    model.train() # model already explicitly moved to device
    for epoch in range(num_epochs): # for each epoch
        print(f'EPOCH {epoch} / {num_epochs}')
        total_loss = 0
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            
            # print('DISTRIBUTION:', (LR_targets == 0).sum(), (LR_targets == 1).sum())
            # print('BATCH_MULTI_EMB', batch_multi_emb)
            w1 = LR_targets.shape[0] / (LR_targets == 0).sum() if (LR_targets == 0).sum() > 0 else 0
            w2 = LR_targets.shape[0] / (LR_targets == 1).sum() if (LR_targets == 1).sum() > 0 else 0
            criterion.weight = torch.tensor([w1, w2], device=device, dtype=torch.float)

            # criterion.weight = torch.tensor([(LR_targets == 0).sum() / LR_targets.shape[0], (LR_targets == 1).sum() / LR_targets.shape[0]], device=device)
            optimizer.zero_grad()
            outputs = model(batch_multi_emb)
            loss = criterion(outputs, LR_targets)
            loss = loss.mean()
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()

            # print(outputs, LR_targets)

        print(f'total loss {epoch}: {total_loss}')

def update_misclassified(train_data, train_emb_loader, model, threshold):
    print('UPDATING MISCLASSIFIED')
    model.eval()
    with torch.no_grad():
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            outputs = model(batch_multi_emb)
            predicted = (outputs[:, 1] > 0.5).long() # if the probability of positive class (1) is greater than 0.5, mark me as 1, else 0
            misclassified = predicted != LR_targets # 1 if misclassified and 0 if correct
            distances = torch.abs(torch.max(outputs, dim=1)[0] - 0.5) # see how far each output is from the decision boundary
            min_distance = torch.min(distances)
            max_distance = torch.max(distances)
            threshold = (max_distance - min_distance) / 2
            print('THRESHOLD:', threshold)
            indices_to_update = torch.nonzero((distances < threshold) & (misclassified), as_tuple=False).squeeze() # get the indices for the distances that are within some threshold of the decision boundary

            train_data.update_LR_y(indices_to_update, 1 - LR_targets[indices_to_update])

def eval_classifier(device, model, val_emb_loader):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():
        correct = 0
        total = 0

        for batch in val_emb_loader:
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            erm_predicted_labels = torch.argmax(batch['embeddings'][str(0)], dim=1)
            true_labels = batch['actual_targets']

            outputs = model(batch_multi_emb)

            all_ones = torch.ones(true_labels.size(0), device=device)
            all_zeros = torch.zeros(true_labels.size(0), device=device)
            
            LR_predicted = torch.argmax(outputs, dim=1)

            tp += ((erm_predicted_labels != true_labels) & (LR_predicted == all_ones)).sum().item()  # how many ERM misclassified points does the classifier put in group 1
            fp += ((erm_predicted_labels == true_labels) & (LR_predicted == all_ones)).sum().item()  # how many ERM correctly classified points does the classifier put in group 1
            tn += ((erm_predicted_labels == true_labels) & (LR_predicted == all_zeros)).sum().item() # how many ERM misclassified points does the classifier put in group 0
            fn += ((erm_predicted_labels != true_labels) & (LR_predicted == all_zeros)).sum().item() # how many ERM correctly classified points does the classifier put in group 0

            total += LR_predicted.size(0)
            correct += tp + tn

        accuracy = correct / total
        print('TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn)
        print(f'Val Accuracy: {accuracy:.4f}')
        ppv = tp/(max(1, tp+fp))
        print(f'TPR: {tp/(max(1, tp+fn))}\tFPR: {fp/(max(1, tn+fp))}\tTNR: {tn/(max(1, tn+fp))}\tFNR: {fn/(max(1, tp+fn))}\tPPV: {ppv}\t1 - PPV: {1 - ppv}')

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


def _find_groups(data, num_epochs, k, groups, test=True):

    group_num = 1 if (len(groups) == 0) else len(groups[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pos_group_count = 0

    train_emb_loader = data['train_loader']
    val_emb_loader = data['val_loader']
    test_emb_loader = data['test_loader']

    if not test:
        print('CREATING LR B_0, K_0 TARGETS')
        for loader, dataset in zip([train_emb_loader, val_emb_loader], [data['train_data'], data['val_data']]): # we don't actually use test_data
            for batch in loader:
                data_idx = batch['idx']
                misclassified = batch['LR_targets']
                dataset.update_LR_y(data_idx, misclassified)

    ex_batch = next(iter(train_emb_loader))['embeddings'] # example batch of embeddings (this is a dictionary)
    ex_batch_multi_emb = reshape_batch_embs(ex_batch) # concatenate embeddings from multiple layers together
    multi_emb_size = ex_batch_multi_emb.shape[-1] # get size of single vector in batch
 
    model = LogisticRegressionModel(multi_emb_size, 2) # binary classifier
    model.to(device) # move LR model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if test:
        for i in range(4):
            find_existing_groups(device, model, train_emb_loader, val_emb_loader, criterion, optimizer, num_epochs, i)
    else:
        train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs)

        eval_classifier(device, model, val_emb_loader)
        # visualize_classifier(train_emb_loader, layer=1, model=model, device=device)

        for i in range(k):
            print(f'SINGLE GROUP FINDING ITER {i}/{k}')
            update_misclassified(data['train_data'], uni_train_emb_loader, model, threshold=1.5)
            train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs)

            eval_classifier(device, model, val_emb_loader)
            visualize_classifier(train_emb_loader, layer=1, model=model, device=device)

            for i in range(k):
                print(f'SINGLE GROUP FINDING ITER {i}/{k}')
                update_misclassified(data['train_data'], train_emb_loader, model, threshold=1.5)
                train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs)

                eval_classifier(device, model, val_emb_loader)
                # visualize_classifier()

    model.eval()
    with torch.no_grad():
        for loader in [train_emb_loader, val_emb_loader]:
            for batch in loader:
                batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
                LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
                outputs = model(batch_multi_emb)
                predicted = (outputs[:, 1] > 0.5).long() # if the probability of positive class (1) is greater than 0.5, mark me as 1, else 0

                data_idx = batch['idx']

                for i, idx in enumerate(data_idx):
                    pos_group_count += predicted[i].item()
                    groups[idx].append(group_num * predicted[i].item() - 1 * (not predicted[i].item()))
                    
    return groups, pos_group_count, (len(groups) - pos_group_count)

def find_groups(data, num_epochs=5, k=1, max_iter=1, min_group=100, groups=None, test=True):
    if not groups: groups = defaultdict(lambda: [0])
    else: 
        torch.load(groups)
        groups = {i: row.tolist() for i, row in enumerate(groups)}

    pos_count, neg_count = float('inf'), float('inf')

    run = 0
    while pos_count > min_group and neg_count > min_group and run < max_iter:
        print('FINDING ONE MORE GROUP')
        groups, pos_count, neg_count = _find_groups(data, num_epochs, k, groups, test)
        run += 1

    save_idx = list(groups.keys())
    save_idx.sort()
    store_groups = [torch.tensor(groups[i]) for i in save_idx]
    store_groups = torch.stack(store_groups)

    torch.save({'group_array': store_groups}, 'classifier_groups.pt')

    
    return groups


# for testing purposes
def find_existing_groups(device, model, train_emb_loader, val_emb_loader, criterion, optimizer, num_epochs, classifier_group=1):
    print('TRAINING LR CLASSIFIER')
    model.train() # model already explicitly moved to device
    for epoch in range(num_epochs): # for each epoch
        print(f'EPOCH {epoch} / {num_epochs}')
        total_loss = 0
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            group_tensor = torch.tensor([classifier_group * batch_multi_emb.shape[0]])
            erm_predicted_labels = torch.argmax(batch['embeddings'][str(0)], dim=1)
            LR_targets = (group_tensor == batch['group']) & (batch['actual_targets'] != erm_predicted_labels) # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU

            # criterion.weight = torch.tensor([(LR_targets == 0).sum() / LR_targets.shape[0], (LR_targets == 1).sum() / LR_targets.shape[0]], device=device)
            optimizer.zero_grad()
            outputs = model(batch_multi_emb)
            loss = criterion(outputs, LR_targets)
            loss = loss.mean()
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()

            # print(outputs, LR_targets)

        print(f'total loss {epoch}: {total_loss}')
        eval_classifier(device, model, val_emb_loader)






    


    
