import gc
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from functools import partial
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
            self.X, self.Y_Array = self._load_samples()
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

def get_model_and_dataset_celeba(args, layer_num):
    dataset = CelebADataset(root='celebA/data', split='train', test_celeba=args.test_celebA)
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
    train_dataset = CelebADataset(root='celebA/data', split='train', test_celeba=True)
    val_dataset = CelebADataset(root='celebA/data', split='val', test_celeba=True)
    test_dataset = CelebADataset(root='celebA/data', split='test', test_celeba=True)
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = create_dataloader(model, datasets, shared_dl_args, layer_num)
    return dataloaders, model, dataset, shared_dl_args, num_classes

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
    data = {'embeddings': embeddings, 'loss': loss, 'predicted_label': pred, 'actual_label': labels, 'class_label': class_labels, 'idxs': idxs}
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
    