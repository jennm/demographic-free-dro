import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

from fair_k_pca import get_fair_subspace


class LinearModel(nn.Module):
    def __init__(self, d, num_classes):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    

def unpack_data(part):
    train_npz = np.load(f'cmnist_meta_{part}.npz')
    predictions = train_npz['predictions']
    embeddings = train_npz['embeddings']
    subclasses = train_npz['subclass']
    labels = train_npz['label']
    data_idx = train_npz['data_idx']
    losses = train_npz['loss']
    
    return predictions, embeddings, subclasses, labels, data_idx, losses


def train(model, num_epochs, trainloader, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')


def evaluate(model, embeddings, labels, subclasses, original_predictions):
    model.eval()
    with torch.no_grad():
        outputs = model(embeddings).squeeze()
        _, predictions = torch.max(outputs, 1)
    
    _, og_predictions = torch.max(original_predictions, 1)
    print('Overall Accuracy:', accuracy_score(labels, predictions), 'Original Overall Accuracy:', accuracy_score(labels, og_predictions))

    group_acc = []

    for i in range(25):
        group_embeddings = embeddings[subclasses == i]
        group_labels = labels[subclasses == i]

        with torch.no_grad():
            group_outputs = model(group_embeddings).squeeze()
            _, group_predictions = torch.max(group_outputs, 1)
        
        group_acc.append(accuracy_score(group_labels, group_predictions))
        _, og_predictions = torch.max(original_predictions[subclasses == i], 1)
        print(f'Group {i}: Accuracy:', group_acc[-1], 'Original Acc:', accuracy_score(group_labels, og_predictions))

    wid = np.argmin(np.array(group_acc))
    print('Worst Group Id:', wid, 'Acc:', group_acc[wid])


def main():
    train_predictions, train_embeddings, train_subclasses, train_labels, _, _ = unpack_data('train_last_epoch')
    # subspace = get_fair_subspace(train_embeddings, train_subclasses, 10)
    # train_embeddings = train_embeddings.dot(subspace)
    # train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    # train_labels = torch.tensor(train_labels, dtype=torch.long)
    # train_predictions = torch.tensor(train_predictions, dtype=torch.long)

    # _, val_embeddings, val_subclasses, val_labels, _, _ = unpack_data('val_last_epoch')
    # # val_embeddings = val_embeddings.dot(subspace)
    # val_embeddings = torch.tensor(val_embeddings, dtype=torch.float32)
    # val_labels = torch.tensor(val_labels, dtype=torch.long)


    # test_predictions, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('test_last_epoch')
    # test_embeddings = test_embeddings.dot(subspace)
    # test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    # test_labels = torch.tensor(test_labels, dtype=torch.long)
    # test_predictions = torch.tensor(test_predictions, dtype=torch.long)

    # d = train_embeddings.shape[1]
    # model = LinearModel(d, 2)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # batch_size = 32

    # train_dataset = TensorDataset(train_embeddings, train_labels)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # train(model, 5, train_loader, optimizer, criterion)
    # evaluate(model, train_embeddings, train_labels, train_subclasses, train_predictions)

    train_predictions = np.argmax(train_predictions, axis=1)
    print('misclassified count:', (train_predictions != train_labels).sum()) # 805
    print('positive predicted count', train_predictions.sum()) # 10844
    print('positive label count', train_labels.sum()) # 10929


main()

            



