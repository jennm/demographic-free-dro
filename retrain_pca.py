import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from itertools import chain

from fair_k_pca import get_fair_subspace, get_fair_subspace_MANUAL_2_GROUPS, get_fair_subspace_MANUAL_2_GROUPS_COSINE, compute_covariance, get_real_eig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



class LinearModel(nn.Module):
    def __init__(self, d, num_classes):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(d, num_classes)
        # self.relu = nn.ReLU()
        # self.out = nn.Linear(h, num_classes)


    def forward(self, x):
        x = self.fc(x)
        # x = self.relu(x)
        return x
    

def unpack_data(part):
    train_npz = np.load(f'embeddings/{part}.npz')
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

def compute_acc(labels, predictions):
    return torch.sum(predictions == labels).item() / len(labels)

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def evaluate(model, embeddings, labels, subclasses, original_predictions, original_logits):
    model.eval()
    with torch.no_grad():
        outputs = model(embeddings).squeeze()
        predictions = torch.argmax(outputs, 1)
    
    original_logits = F.softmax(original_logits, dim=1)
    outputs = F.softmax(outputs, dim=1)

    combined_logits = torch.cat((outputs, original_logits), dim=1)
    stolen_predictions = torch.argmax(combined_logits, 1) % 2
    print('Overall Accuracy:', compute_acc(labels, predictions), 'Stolen Accuracy:', compute_acc(labels, stolen_predictions), 'Original Overall Accuracy:', compute_acc(labels, original_predictions))

    group_acc = []

    for i in range(len(set(subclasses))):
        group_labels = labels[subclasses == i]
        group_predictions = predictions[subclasses == i]
        original_group_predictions = original_predictions[subclasses == i]


        group_acc.append(compute_acc(group_labels, group_predictions))
        original_group_acc = compute_acc(group_labels, original_group_predictions)

        max_logits, _ = torch.max(outputs[subclasses == i], dim=1)
        max_logits /= torch.sum(outputs[subclasses == i], dim=1)
        average_max_logits = torch.mean(max_logits.float())
        original_max_logits, _ = torch.max(original_logits[subclasses == i], dim=1) 
        original_max_logits /= torch.sum(original_logits[subclasses == i], dim=1)
        original_average_max_logits = torch.mean(original_max_logits.float())

        combined_group_logits = torch.cat((outputs[subclasses == i], original_logits[subclasses == i]), dim=1)
        stolen_group_predictions = torch.argmax(combined_group_logits, 1) % 2
        stolen_group_acc = compute_acc(group_labels, stolen_group_predictions)

        print(f'{Color.RED if original_group_acc > group_acc[-1] else Color.GREEN}Group {i:<5}: Accuracy: {group_acc[-1]:<10.4f} Stolen Accuracy: {stolen_group_acc:<10.4f} Confidence: {average_max_logits.item():<10.4f} Original Acc:{original_group_acc:<10.4f} Original Confidence: {original_average_max_logits.item():<10.4f}{Color.END}')

    wid = np.argmin(np.array(group_acc))
    print('Worst Group Id:', wid, 'Acc:', group_acc[wid])

def compute_projection_all(embeddings, subspace):
    return embeddings.dot(subspace)

def compute_projection(embeddings, subspace, mask):
    relevant_rows = embeddings[mask]
    projected_rows = relevant_rows @ subspace
    embeddings[mask] = projected_rows


    return embeddings


def get_subspace(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    # G1 = np.concatenate((embeddings[subclasses == 0][:200], embeddings[subclasses == 1][:200]), axis=0)
    # G2 = np.concatenate((embeddings[subclasses == 2][:200], embeddings[subclasses == 3][:200]), axis=0)
    # G = np.concatenate((G1, G2), axis=0)
    # labels = np.array([0] * 400 + [1] * 400)

    subspace = get_fair_subspace(embeddings, labels, 10, [0, 1])

    # female_brunettes = embeddings[(subclasses == 0)]# & (misclassified == 0)]
    # female_blondes = embeddings[(subclasses == 1)]# & (misclassified == 0)]
    # male_brunettes = embeddings[(subclasses == 2)]# & (misclassified == 0)]
    # male_blondes = embeddings[(subclasses == 3)]# & (misclassified == 1)]

    # N = 200
    # # G1 = np.concatenate((female_brunettes[:N], female_blondes[:N], male_brunettes[:N], male_blondes[:N]), axis=0)
    # G1 = np.concatenate((male_brunettes[:N], male_blondes[:N]), axis=0)


    # # G1 = embeddings
    # A1 = compute_covariance(G1)
    # v, e = get_real_eig(A1)

    # pca1 = e[:, -1].reshape(-1, 1)

    # V = pca1
    # subspace = np.eye(G1.shape[1]) - V @ V.T

    embeddings = embeddings @ subspace
    test_embeddings = test_embeddings @ subspace
    return embeddings, test_embeddings


def scale_G(G):
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    G = G / norms
    noise_scale = 0.0001
    noise = noise_scale * np.random.randn(*G.shape)
    G = G + noise
    return G

def group_accuracy_score(y, y_pred, subclasses):
    return {'Blue 1s': accuracy_score(y[subclasses == 0], y_pred[subclasses == 0]),
    'Red 1s': accuracy_score(y[subclasses == 1], y_pred[subclasses == 1]),
    'Blue 0s': accuracy_score(y[subclasses == 2], y_pred[subclasses == 2]),
    'Red 0s': accuracy_score(y[subclasses == 3], y_pred[subclasses == 3]),
    'Overall Accuracy': accuracy_score(y, y_pred)}


def compare_linear_sep(embeddings, subclasses, test_embeddings, test_subclasses, misclassified):
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

 
def main():
    seed = 42
    torch.manual_seed(seed)

    epoch = 12
    dataset = 'CelebA'
    logits, embeddings, subclasses, labels, _, _ = unpack_data(f'{dataset}_meta_train_{epoch}_epoch')
    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data(f'{dataset}_meta_test_{epoch}_epoch')

    test_predictions = np.argmax(test_logits, axis=1)
    test_misclassified = test_predictions != test_labels

    embeddings = scale_G(embeddings)
    test_embeddings = scale_G(test_embeddings)

    # subclasses[subclasses == 0] = 10
    # subclasses[subclasses == 1] = 12
    # subclasses[subclasses == 2] = 11
    # subclasses[subclasses == 3] = 13
    # subclasses[subclasses == 10] = 0 # female brunettes
    # subclasses[subclasses == 11] = 1 # female blondes
    # subclasses[subclasses == 12] = 2 # male brunettes
    # subclasses[subclasses == 13] = 3 # male blondes

    # test_subclasses[test_subclasses == 0] = 10
    # test_subclasses[test_subclasses == 1] = 12
    # test_subclasses[test_subclasses == 2] = 11
    # test_subclasses[test_subclasses == 3] = 13
    # test_subclasses[test_subclasses == 10] = 0
    # test_subclasses[test_subclasses == 11] = 1
    # test_subclasses[test_subclasses == 12] = 2
    # test_subclasses[test_subclasses == 13] = 3

    ######################################################################################
    # red_1s = np.array([1, 1, 0])
    # red_2s = np.array([1, 0, 1])
    # blue_1s = np.array([-1, 1, 0])
    # blue_2s = np.array([-1, 0, 1])

    # subclasses = np.array([1, 3, 0, 2])
    # labels = np.array([0, 1, 0, 1])
    # embeddings = np.array([red_1s, red_2s, blue_1s, blue_2s])

    # print(embeddings)
    # embeddings, test_embeddings = get_subspace(embeddings, subclasses, None, labels, embeddings, subclasses, None)
    # print(embeddings)
    # exit()
    ######################################################################################

    print(np.linalg.matrix_rank(embeddings))
    compare_linear_sep(embeddings, subclasses, test_embeddings, test_subclasses, misclassified)    
    embeddings, test_embeddings = get_subspace(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified)
    print(np.linalg.matrix_rank(embeddings))
    compare_linear_sep(embeddings, subclasses, test_embeddings, test_subclasses, misclassified)

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_predictions = torch.tensor(test_predictions, dtype=torch.long)

    d = embeddings.shape[1]
    model = LinearModel(d, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    batch_size = 32

    val_dataset = TensorDataset(embeddings, labels)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    for e in range(1):
        print('Epoch Interval:', e * 15)
        train(model, 15, val_loader, optimizer, criterion)
        print('TRAIN EVALUATION')
        evaluate(model, embeddings, labels, subclasses, torch.tensor(predictions, dtype=torch.long), torch.tensor(logits, dtype=torch.float32))
        print('TEST EVALUATION')
        evaluate(model, test_embeddings, test_labels, test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))

# main()

            



