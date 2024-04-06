import torch
from classifier import LogisticRegressionModel
import torch.nn as nn
import torch.optim as optim

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

def train_classifier(model, train_emb_loader, criterion, optimizer, num_epochs):
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

def update_misclassified():
    model.eval()
    with torch.no_grad():
        for batch in train_emb_loader: # get each batch
            batch_multi_emb = reshape_batch_embs(batch['embeddings']) # reshape to combine embeddings from multiple layers, already on GPU
            LR_targets = batch['LR_targets'] # B_k, R_k from discussion NOT actual data targets like Male Female, not on GPU
            outputs = model(batch_multi_emb)
            predicted = (outputs[:, 1] > 0.5).long() # if the probability of positive class (1) is greater than 0.5, mark me as 1, else 0
            misclassified = predicted != LR_targets # 1 if misclassified and 0 if correct
            distances = torch.abs(outputs - 0.5) # see how far each output is from the decision boundary
            indices_to_update = torch.nonzero((distances < threshold) & (misclassified), as_tuple=False).squeeze() # get the indices for the distances that are within some threshold of the decision boundary

            train_emb_loader.upda




def find_groups(data, num_epochs, k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    groups = dict()

    train_emb_loader = data['train_loader']
    val_emb_loader = data['val_loader']
    test_emb_loader = data['test_loader']

    ex_batch = next(iter(train_emb_loader))['embeddings'] # example batch of embeddings (this is a dictionary)
    ex_batch_multi_emb = reshape_batch_embs(ex_batch) # concatenate embeddings from multiple layers together
    multi_emb_size = ex_batch_multi_emb.shape[-1] # get size of single vector in batch
 
    model = LogisticRegressionModel(multi_emb_size, 2) # binary classifier
    model.to(device) # move LR model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_classifier(model, train_emb_loader, criterion, optimizer, num_epochs)
    # eval_classifier()
    # visualize_classifier()

    for i in range(k):
        update_misclassified()
        train_model(model, train_emb_loader, criterion, optimizer, num_epochs)
        eval_classifier()
        visualize_classifier()
    
    return groups

