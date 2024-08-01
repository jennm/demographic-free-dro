import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from itertools import chain

from fair_k_pca import get_fair_subspace, get_fair_subspace_MANUAL_2_GROUPS, get_fair_subspace_MANUAL_2_GROUPS_COSINE, compute_covariance, get_real_eig, get_fair_subspace_MANUAL_N_GROUPS_COSINE, get_fair_subspace_MANUAL_N_GROUPS
from encoder_decoder import EncoderDecoder, train_encoder_decoder, CosineSimilarityLoss
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler


class LinearModel(nn.Module):
    def __init__(self, d, num_classes):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
    

def unpack_data(part):
    train_npz = np.load(f'{part}.npz')
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

def subspace_per_group(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    zero_subspace = get_fair_subspace(embeddings, subclasses, 50, range(0, 5))
    one_subspace = get_fair_subspace(embeddings, subclasses, 50, range(5, 10))
    two_subspace = get_fair_subspace(embeddings, subclasses, 50, range(10, 15))
    three_subspace = get_fair_subspace(embeddings, subclasses, 50, range(15, 20))
    four_subspace = get_fair_subspace(embeddings, subclasses, 50, range(20, 25))

    original_embeddings = embeddings.copy()

    embeddings = compute_projection(embeddings, zero_subspace, np.isin(subclasses, list(range(0, 5))))
    embeddings = compute_projection(embeddings, one_subspace, np.isin(subclasses, list(range(5, 10))))
    embeddings = compute_projection(embeddings, two_subspace, np.isin(subclasses, list(range(10, 15))))
    embeddings = compute_projection(embeddings, three_subspace, np.isin(subclasses, list(range(15, 20))))
    embeddings = compute_projection(embeddings, four_subspace, np.isin(subclasses, list(range(20, 25))))

    # test_embeddings = compute_projection(test_embeddings, zero_subspace, np.isin(test_subclasses, list(range(0, 5))))
    # test_embeddings = compute_projection(test_embeddings, one_subspace, np.isin(test_subclasses, list(range(5, 10))))
    # test_embeddings = compute_projection(test_embeddings, two_subspace, np.isin(test_subclasses, list(range(10, 15))))
    # test_embeddings = compute_projection(test_embeddings, three_subspace, np.isin(test_subclasses, list(range(15, 20))))
    # test_embeddings = compute_projection(test_embeddings, four_subspace, np.isin(test_subclasses, list(range(20, 25))))

    input_dim = embeddings.shape[1]
    model = EncoderDecoder(input_dim)

    class_counts = np.bincount(misclassified)
    class_weights = 1. / class_counts
    sample_weights = class_weights[misclassified.astype(np.int64)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    model = train_encoder_decoder(model, 
                          DataLoader(
                              TensorDataset(
                                  torch.tensor(original_embeddings, dtype=torch.float32), 
                                  torch.tensor(embeddings, dtype=torch.float32)
                            ), 
                            batch_size=32, 
                            sampler=sampler
                        ),
                        nn.MSELoss(),
                        optim.Adam(model.parameters(), lr=0.001)
            )

    # test_embeddings = compute_projection(test_embeddings, positive_subspace, np.isin(test_subclasses, list(range(15, 20))))
    # test_embeddings = compute_projection(test_embeddings, negative_subspace, np.isin(test_subclasses, list(chain(range(0, 15), range(20, 25)))))
    test_embeddings = model(torch.tensor(test_embeddings, dtype=torch.float32))

    return embeddings, test_embeddings

def subspace_per_label(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    positive_subspace = get_fair_subspace(embeddings, subclasses, 50, range(15, 20))
    negative_subspace = get_fair_subspace(embeddings, subclasses, 50, chain(range(0, 15), range(20, 25)))

    original_embeddings = embeddings.copy()

    embeddings = compute_projection(embeddings, positive_subspace, np.isin(subclasses, list(range(15, 20))))
    embeddings = compute_projection(embeddings, negative_subspace, np.isin(subclasses, list(chain(range(0, 15), range(20, 25)))))

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    input_dim = embeddings.shape[1]
    model = EncoderDecoder(input_dim)

    class_counts = np.bincount(misclassified)
    class_weights = 1. / class_counts
    sample_weights = class_weights[(misclassified).astype(np.int64)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    model = train_encoder_decoder(model, 
                          DataLoader(
                              TensorDataset(
                                  torch.tensor(original_embeddings, dtype=torch.float32), 
                                  torch.tensor(embeddings, dtype=torch.float32)
                            ), 
                            batch_size=32, 
                            # shuffle=True,
                            sampler=sampler
                        ),
                        CosineSimilarityLoss(),
                        optim.Adam(model.parameters(), lr=0.001)
            )

    # test_embeddings = compute_projection(test_embeddings, positive_subspace, np.isin(test_subclasses, list(range(15, 20))))
    # test_embeddings = compute_projection(test_embeddings, negative_subspace, np.isin(test_subclasses, list(chain(range(0, 15), range(20, 25)))))
    test_embeddings = model(torch.tensor(test_embeddings, dtype=torch.float32))

    scaler = StandardScaler()
    test_embeddings = scaler.fit_transform(test_embeddings.detach().numpy())

    return embeddings, test_embeddings

def subspace_per_label_ADD(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    positive_subspace = get_fair_subspace(embeddings[misclassified], subclasses[misclassified], 50, range(len(set(subclasses))))
    negative_subspace = get_fair_subspace(embeddings[~misclassified], subclasses[~misclassified], 50, range(len(set(subclasses))))

    combined_subspace = positive_subspace + negative_subspace

    embeddings = compute_projection_all(embeddings, combined_subspace)
    # test_embeddings = compute_projection_all(test_embeddings, combined_subspace)

    # embeddings = compute_projection(embeddings, positive_subspace, misclassified)
    # embeddings = compute_projection(embeddings, negative_subspace, ~misclassified)



    # scaler = StandardScaler()
    # embeddings = scaler.fit_transform(embeddings)

    # A = compute_covariance(embeddings)
    # eigenvalues, eigenvectors = get_real_eig(A)

    # D = len(eigenvectors[0])
    # P = np.zeros((D, D), dtype='float')

    # for idx in range(30):
    #     u_i = eigenvectors[:, -idx].reshape((D, 1))
    #     P += (u_i @ u_i.T)

    # test_embeddings = compute_projection_all(test_embeddings, P)

    return embeddings, test_embeddings

def one_subspace(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    # answes the question: does fpca cut out redness vs antiredness
    subspace = get_fair_subspace(embeddings, subclasses, 77, range(2), True, np.isin(subclasses, list(range(3, 25, 5))))
    # subspace = get_fair_subspace(embeddings, labels, 70, range(2))

    embeddings = compute_projection_all(embeddings, subspace)
    # embeddings = compute_projection(embeddings, subspace, misclassified)
    test_embeddings = compute_projection_all(test_embeddings, subspace)

    return embeddings, test_embeddings

def one_subspace_HACKY_2_GROUPS(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    subspace = get_fair_subspace_MANUAL_2_GROUPS(1, 3, embeddings, subclasses)

    embeddings = compute_projection_all(embeddings, subspace)
    # embeddings = compute_projection(embeddings, subspace, misclassified)
    test_embeddings = compute_projection_all(test_embeddings, subspace)
    # test_embeddings = compute_projection(test_embeddings, subspace, np.isin(test_subclasses, list(range(15, 20))))

    return embeddings, test_embeddings

def one_subspace_HACKY_N_GROUPS(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    subspace = get_fair_subspace_MANUAL_N_GROUPS(list(range(25)), embeddings, subclasses)

    embeddings = compute_projection_all(embeddings, subspace)
    # embeddings = compute_projection(embeddings, subspace, misclassified)
    test_embeddings = compute_projection_all(test_embeddings, subspace)
    # test_embeddings = compute_projection(test_embeddings, subspace, np.isin(test_subclasses, list(range(15, 20))))

    return embeddings, test_embeddings

def one_subspace_HACKY_2_GROUPS_COSINE(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    subspace = get_fair_subspace_MANUAL_2_GROUPS_COSINE(1, 3, embeddings, subclasses)

    embeddings = compute_projection_all(embeddings, subspace)
    # embeddings = compute_projection(embeddings, subspace, np.isin(subclasses, list(range(15, 20))))
    test_embeddings = compute_projection_all(test_embeddings, subspace)
    # test_embeddings = compute_projection(test_embeddings, subspace, np.isin(test_subclasses, list(range(15, 20))))

    return embeddings, test_embeddings

def one_subspace_HACKY_N_GROUPS_COSINE(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    subspace = get_fair_subspace_MANUAL_N_GROUPS_COSINE(list(range(len(set(subclasses)))), embeddings, subclasses)

    embeddings = compute_projection_all(embeddings, subspace)
    # embeddings = compute_projection(embeddings, subspace, np.isin(subclasses, list(range(15, 20))))
    test_embeddings = compute_projection_all(test_embeddings, subspace)
    # test_embeddings = compute_projection(test_embeddings, subspace, np.isin(test_subclasses, list(range(15, 20))))

    return embeddings, test_embeddings

def one_subspace_POSITIVE_LABELS(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified):
    subspace = get_fair_subspace(embeddings, subclasses, 50, range(15, 20))

    embeddings = compute_projection(embeddings, subspace, np.isin(subclasses, list(range(15, 20))))
    test_embeddings = compute_projection_all(test_embeddings, subspace)

    return embeddings, test_embeddings

def main():
    seed = 42
    torch.manual_seed(seed)

    logits, embeddings, subclasses, labels, _, _ = unpack_data('cmnist_meta_train_17_epoch') # unpack_data('train_17_epoch')
    predictions = np.argmax(logits, axis=1)
    misclassified = predictions != labels

    test_logits, test_embeddings, test_subclasses, test_labels, _, _ = unpack_data('cmnist_meta_test_17_epoch') # unpack_data('test_17_epoch')

    test_predictions = np.argmax(test_logits, axis=1)
    test_misclassified = test_predictions != test_labels

    embeddings, test_embeddings = one_subspace(embeddings, subclasses, misclassified, labels, test_embeddings, test_subclasses, test_misclassified)

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_predictions = torch.tensor(test_predictions, dtype=torch.long)

    d = embeddings.shape[1]
    model = LinearModel(d, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch_size = 32

    val_dataset = TensorDataset(embeddings, labels)

    # misclassified_count = misclassified.sum()
    # correct_count = len(misclassified) - misclassified_count
    # correct_wrong_weights = [len(misclassified) / correct_count, len(misclassified) / misclassified_count]
    # weights = correct_wrong_weights[misclassified]

    # misclassified = subclasses
    # # misclassified = labels
    # misclassified = torch.tensor(misclassified, dtype=torch.long)
    # class_counts = torch.bincount(misclassified)
    # class_weights = 1. / class_counts.float()
    # # class_weights = torch.tensor([0.2, 0.8], dtype=torch.float)
    # weights = class_weights[misclassified]
    # sampler = WeightedRandomSampler(weights, len(misclassified), replacement=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True) # sampler=sampler)

    for e in range(10):
        print('Epoch Interval:', e * 5)
        train(model, 5, val_loader, optimizer, criterion)
        evaluate(model, test_embeddings, test_labels, test_subclasses, test_predictions, torch.tensor(test_logits, dtype=torch.float32))

main()

            



'''
learn subspace that deletes colors keeps rest
find trick to vis stuff
'''