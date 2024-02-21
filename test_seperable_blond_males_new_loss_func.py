import argparse
import gc
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from data.data import dataset_attributes
from functools import partial
from classifier import LogisticRegressionModel
from models import model_attributes
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import get_model




class CelebADataset(Dataset):
    _normalization_stats = {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}

    def __init__(self, root, split, transform=None, test_celeba=False):
        # Initialize your dataset
        # Load your dataset from the CSV file or any other source
        self.root = root
        self.split = split
        self.test_celeba = test_celeba
        logging.info(f'Loading {self.split} split of CelebA')
        if self.test_celeba:
            self.X, self.Y_Array, self.class_labels = self._load_samples()
        else:
            self.X, self.Y_Array
        self.transform = get_transform_celebA()
        
    def get_num_classes(self):
        return torch.max(self.Y_Array).item() + 1

    def _load_samples(self):
        self.target_name = 'Blond_Hair'
        
        attrs_df = pd.read_csv(os.path.join(self.root, 'list_attr_celeba.csv'),
                               delim_whitespace=True)

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root, 'img_align_celeba')
        filename_array = attrs_df['image_id'].values
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attr_names = attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        def attr_idx(attr_name):
            return attr_names.get_loc(attr_name)

        # Get the y values
        target_idx = attr_idx(self.target_name)
        y_array = attrs_df[:, target_idx]
        self._num_supclasses = np.amax(y_array).item() + 1

        # Read in train/val/test splits
        split_df = pd.read_csv(os.path.join(self.root, 'list_eval_partition.csv'),
                               delim_whitespace=True)
        split_array = split_df['partition'].values
        split_dict = {'train': 0, 'val': 1, 'test': 2}
        split_indices = split_array == split_dict[self.split]
        self.len_dataset = len(split_indices) 

        X = filename_array[split_indices]
        y_array = torch.tensor(y_array[split_indices], dtype=torch.float)
        self.len_dataset = y_array.shape[0] 

        if self.test_celeba:
            male_idx = attr_idx('Male')
            class_labels = list()
            for i in range(attrs_df.shape[0]):
                if attrs_df[i][target_idx] == 1 and attrs_df[i][male_idx] == 1:
                    class_labels.append(1)
                else:
                    class_labels.append(0)
            class_labels = torch.tensor(class_labels, dtype=torch.float)
            return X, y_array, class_labels

        return X, y_array
        

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (x_dict, y_dict) where x_dict is a dictionary mapping all
                possible inputs and y_dict is a dictionary for all possible labels.
        """
        img_filename = os.path.join(self.data_dir, self.X[idx])
        image = Image.open(img_filename)
        if self.transform is not None:
            image = self.transform(image)
        x = image
        x = x.to(torch.float)
        img_idx = int(self.X[idx][0:-4])
        if self.test_celeba:
            return {'image': x, 'label': self.Y_Array[idx], 'class_labels': self.class_labels[idx], 'idx': img_idx}
        return {'image': x, 'label': self.Y_Array[idx], 'idx': img_idx}

def get_transform_celebA():
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize(**CelebADataset._normalization_stats),
    ])
    return transform

tail_cache = dict()
def hook_fn(module, input, output):
    device = output.get_device()
    if device in tail_cache:
        tail_cache[device].append(input[0].clone().detach())
    else:
        tail_cache[device] = [input[0].clone().detach()]
    

def get_hooks(model):
    hooks = []
    num_layers = sum(1 for _ in model.modules())
    print('model num layers', num_layers)
    for i, module in enumerate(model.modules()):
        if i >= num_layers - 5:
            print(f"{i}: {num_layers - i}")
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    print('hooks_length: ', len(hooks))

def setup(args):
    dataset = CelebADataset(root='celebA/data', split='train', test_celeba=True)
    num_classes = int(dataset.get_num_classes())
    model = get_model(
        model=args.model,
        pretrained=not args.train_from_scratch,
        resume=True,
        n_classes=num_classes,
        dataset=args.dataset,
        log_dir=args.log_dir,
    )
    get_hooks(model)
    shared_dl_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    return model, dataset, shared_dl_args, num_classes

def collate_func(batch, pretrained_model, criterion, layer_num):
    global tail_cache
    inputs = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    class_labels = torch.stack([item['class_labels'] for item in batch])
    idxs = [item['idx'] for item in batch]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        # Forward pass through the 5th to last layer
        inputs = inputs.to(device)
        outputs = pretrained_model(inputs)
        pred = torch.argmax(outputs, 1)
        # loss per batch for now
        pred = pred.to(torch.float)
        labels = labels.to(device)
        loss = criterion(pred, labels)


    embeddings = torch.cat(tuple([tail_cache[i][layer_num + 1].to(device) for i in list(tail_cache.keys())]), dim=0)
    data = {'embeddings': embeddings, 'loss': loss, 'predicted_label': labels, 'class_label': class_labels, 'idxs': idxs}
    tail_cache = dict()
    gc.collect()
    torch.cuda.empty_cache()

    return data

def create_dataloader(model, datasets, shared_dl_args, layer_num=1, criterion=nn.CrossEntropyLoss(reduction='none')):
    if type(datasets) is dict:
        dataloaders = dict()
        for dataset_type in datasets:
                collate_fn = partial(collate_func, pretrained_model=model, criterion=criterion, layer_num=layer_num)
                dataloaders[dataset_type] = DataLoader(datasets[dataset_type], **shared_dl_args, collate_fn=collate_fn)
    else:
        # Create a DataLoader
        dataloaders = DataLoader(datasets, **shared_dl_args, collate_fn=collate_fn)

    return dataloaders
    

def train_test_classifier(args):
    old_model, dataset, shared_dl_args, num_classes = setup(args)
    


    # Set random seed for reproducibility
    torch.manual_seed(42)
    mp.set_start_method('spawn')

    # Define transformations and download MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CelebADataset(root='celebA/data', split='train', test_celeba=True)
    val_dataset = CelebADataset(root='celebA/data', split='val', test_celeba=True)
    test_dataset = CelebADataset(root='celebA/data', split='test', test_celeba=True)
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    # data_to_save = {'input_size': [], 'num_classes': [], 'classifiers': []}
    groups_from_classifiers = dict()
    group_counts = list()
    

    dataloaders = create_dataloader(old_model, datasets, shared_dl_args, 3)
    for data_type in ['train', 'test', 'val']:
        for batch in dataloaders[data_type]:
            idxs = batch['idxs']
            for idx in idxs:
                groups_from_classifiers[idx] = list()
    
    group_num = 1
    count = 0

    
    for i in range(5):
        print(f'Layer {i}')
        dataloaders = create_dataloader(old_model, datasets, shared_dl_args, i)
        first_batch_embeddings = next(iter(dataloaders['train']))['embeddings']
        first_batch_embeddings = first_batch_embeddings.view(first_batch_embeddings.size(0), -1)


        # Initialize the model, loss function, and optimizer
        input_size = first_batch_embeddings.shape[-1]  # MNIST images are 28x28 pixels
        num_classes = 2  # use a variable
        log_model = LogisticRegressionModel(input_size, num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.01,1], device=torch.cuda.current_device()), reduction='none')
        optimizer = optim.SGD(log_model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 5

        for epoch in range(num_epochs):
            log_model.train()
            for batch in dataloaders['train']:
                device = torch.cuda.current_device()
                log_model.to(device)
                embeddings = batch['embeddings']
                # loss = batch['loss']
                class_labels = batch['class_label']

                # Flatten the images
                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                class_labels = class_labels.to(device)

                # Forward pass
                outputs = log_model(embeddings)

                # Changing torch type
                class_labels = class_labels.to(torch.long)

                # Calculate loss
                loss =  criterion(outputs, class_labels)
                loss_mean = torch.mean(loss)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

            # Print training loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_mean.item():.4f}')
            gc.collect()
            torch.cuda.empty_cache()


        # Evaluation on the test set
        log_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in dataloaders['test']:
                embeddings = batch['embeddings']
                loss = batch['loss']
                class_labels = batch['class_label']
                class_labels = class_labels.to(device)
                class_labels = class_labels.to(torch.long)

                embeddings = embeddings.view(embeddings.size(0), -1)
                outputs = log_model(embeddings)

                _, predicted = torch.max(outputs, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()

            accuracy = correct / total
            print(f'Test Accuracy: {accuracy:.4f}')

            gc.collect()
            torch.cuda.empty_cache()

            count = 0
            for data_type in ['train', 'test', 'val']:
                print(group_num, data_type)
                for batch in dataloaders[data_type]:
                    embeddings = batch['embeddings']
                    idxs = batch['idxs']
                    embeddings = embeddings.view(embeddings.size(0), -1)
                    outputs = log_model(embeddings)
                    _, predicted = torch.max(outputs, 1)
                    for i in range(len(idxs)):
                        idx = idxs[i]
                        if predicted[i].item() == 1:
                            groups_from_classifiers[idx].append(group_num)
                            count += 1
                        else:
                            groups_from_classifiers[idx].append(-1)
                gc.collect()
                torch.cuda.empty_cache()
            group_counts.append(count)
            group_num += 1
            count = 0

            gc.collect()
            torch.cuda.empty_cache()





        # data_to_save['input_size'].append(input_size)
        # data_to_save['num_classes'].append(num_classes)
        # data_to_save['classifiers'].append(log_model.state_dict())

    example_idxs = list(groups_from_classifiers.keys())
    example_idxs.sort()
    groups_from_classifiers_list = list()
    for idx in example_idxs:
        groups_from_classifiers_list.append(groups_from_classifiers[idx])
    groups_from_classifiers_tensor = torch.tensor(groups_from_classifiers_list)

    # torch.save(data_to_save, "classifiers.pt")
    torch.save({'group_array': groups_from_classifiers_tensor, 'group_counts': torch.tensor(group_counts)}, "groups_from_classifiers_info.pt")

    return log_model, accuracy






def list_of_columns(arg):
    arg.split(' ')

def find_poor_performing_group_indices(groups, data_info_path):
    indices = list()
    data_info_df = pd.read_csv(data_info_path)
    cols = list(data_info_df.columns)[0].split(' ')
    relevant_cols = list()
    i = 0
    while len(relevant_cols) < len(groups) and i < len(cols):
        if cols[i] in groups:
            relevant_cols.append(i)
    
    for index, row in data_info_df.iterrows():
        all_relevant = True
        for j in range(relevant_cols):
            if row.iloc[j] != 1:
                all_relevant = False
                break
        if all_relevant:
            indices.append(index)

    return indices




def main():
    parser = argparse.ArgumentParser(description='tests that poor performinc class is separable by embeddings')
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        default="celebA")
    parser.add_argument("--model",
                        choices=model_attributes.keys(),
                        default="resnet50")
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    log_model, acc = train_test_classifier(args)
    print('Accuracy:', acc)
    

    



if __name__ == '__main__':
    main()
