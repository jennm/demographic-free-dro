import torch
from classifier import LogisticRegressionModel
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict

'''
NOTE: The index that the DataLoader uses when getting a datapoint is the index relative to a fixed, static underlying data store.
Shuffling just changes the order in which indices are accessed. Split just changes the order in which indices are split.
Subset takes in a subset of indices from the full dataset and indexes relative to the full dataset. It does not do any reindexing.
'''

def reshape_batch_embs(batch):
    layers = batch.keys() # batch is really a dictionary of layer embeddings, get keys for all layers stored
    batch_multi_emb = [] # store flattened embeddings from each layer
    for layer in layers: # for each layer
        layer_emb = batch[layer] # get the layer embeddings
        batch_multi_emb.append(layer_emb.view(layer_emb.size(0), -1)) # append the flattened embeddings i.e. (32 x 10 x 10 => 32 x 100)
    return torch.cat(batch_multi_emb, dim=1) # concatenate the flattened embeddings i.e. (32 x 10 and 32 x 20 => 32 x 30)

def train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs):
    model.train() # model already explicitly moved to device
    for epoch in range(num_epochs): # for each epoch
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU

            criterion.weight = torch.tensor([(LR_targets == 0).sum() / LR_targets.shape[0], (LR_targets == 1).sum() / LR_targets.shape[0]], device=device)
            optimizer.zero_grad()
            outputs = model(batch_multi_emb)
            loss = criterion(outputs, LR_targets)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

def update_misclassified(train_data, train_emb_loader, model, threshold):
    model.eval()
    with torch.no_grad():
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            outputs = model(batch_multi_emb)
            predicted = (outputs[:, 1] > 0.5).long() # if the probability of positive class (1) is greater than 0.5, mark me as 1, else 0
            misclassified = predicted != LR_targets # 1 if misclassified and 0 if correct
            distances = torch.abs(torch.max(outputs, dim=1)[0] - 0.5) # see how far each output is from the decision boundary
            indices_to_update = torch.nonzero((distances < threshold) & (misclassified), as_tuple=False).squeeze() # get the indices for the distances that are within some threshold of the decision boundary

            train_data.update_LR_y(indices_to_update, 1 - LR_targets[indices_to_update])


def _find_groups(data, num_epochs, k, groups):

    group_num = 1 if (len(groups) == 0) else (max(groups.keys()) + 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pos_group_count = 0

    train_emb_loader = data['train_loader']
    val_emb_loader = data['val_loader']
    test_emb_loader = data['test_loader']

    for loader, dataset in zip([train_emb_loader, val_emb_loader, test_emb_loader], [data['train_data'], data['val_data'], data['test_data']]):
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

    train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs)

    # eval_classifier()
    # visualize_classifier()

    for i in range(k):
        update_misclassified(data['train_data'], train_emb_loader, model, threshold=1.5)
        train_classifier(device, model, train_emb_loader, criterion, optimizer, num_epochs)

        # eval_classifier()
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

def find_groups(data, num_epochs=1, k=1, max_iter=1, min_group=10, groups=None):
    if not groups: groups = defaultdict(lambda: [0])
    else: 
        torch.load(groups)
        groups = {i: row.tolist() for i, row in enumerate(groups)}

    pos_count, neg_count = float('inf'), float('inf')

    run = 0
    while pos_count > min_group and neg_count > min_group and run < max_iter:
        groups, pos_count, neg_count = _find_groups(data, num_epochs, k, groups)
        run += 1

    save_idx = list(groups.keys())
    save_idx.sort()
    store_groups = [torch.tensor(groups[i]) for i in save_idx]
    store_groups = torch.stack(store_groups)

    torch.save({'group_array': store_groups}, 'classifier_groups.pt')

    
    return groups




    


    
