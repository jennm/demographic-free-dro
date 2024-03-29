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
from models import model_attributes
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import get_model



class ColoredMNISTDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split,
                 target_name,
                 confounder_names,
                 model_type,
                 augment_data=False,
                 metadata_csv_name="metadata.csv",
                 test_mnist=False):
        
        self.root_dir = os.path.join(root_dir, "coloredMNIST")
        self.target_name = target_name # new class 3
        self.augment_data = augment_data
        self.model_type = model_type
        self.test_mnist = test_mnist

        # read in attributes
        self.attrs_df = pd.read_csv(
            os.path.join(self.root_dir, "data", metadata_csv_name)
        )

        self.data_dir = os.path.join(self.root_dir, "data", "colored_mnist_imgs")
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # should be 0 or 1
        self.attrs_df = self.attrs_df.values
        split_indices = self.attrs_df 

        print('confounder:', confounder_names)
        target_idx = self.attr_idx(self.target_name) # get id for "3"
        # if self.test_mnist:
        #     class_labels = list()
        #     for i in range(self.attrs_df.shape[0]):
        #         if self.attrs_df[i][target_idx] == 1: #and self.attrs_df[i][confounder_idx] == 1
        #             class_labels.append(1)
        #         else:
        #             class_labels.append(0)
        self.y_array = self.attrs_df[:, target_idx]

        # self.n_classes = 2 # i am either 3 or not 3 

        if type(confounder_names) is list:
            self.confounder_idx = [self.attr_idx(a) for a in confounder_names]
        else:
            self.confounder_idx = [self.attr_idx(confounder_names)]
        if self.test_mnist:
            self.class_labels = list()
            for i in range(self.attrs_df.shape[0]):
                for idx in range(len(self.confounder_idx)):
                    if self.attrs_df[i][target_idx] == 1 and self.attrs_df[i][self.confounder_idx[idx]] == 1:
                        self.class_labels.append(idx + 1)
                    else:
                        self.class_labels.append(0)
            
            # print(len(self.class_labels), self.class_labels)
        # self.n_confounders = len(self.confounder_idx)
        # confounders = self.attrs_df[:, self.confounder_idx]
        # self.confounder_array = np.matmul(
            # confounders.astype(int),
            # np.power(2, np.arange(len(self.confounder_idx)))
        # )

        # self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        # self.group_array = (self.y_array * (self.n_groups / 2)
                            # + self.confounder_array).astype("int")
        
        self.split_df = pd.read_csv(
            os.path.join(self.root_dir, "data", metadata_csv_name)
        )
        self.split_array = self.split_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2
        }
        split_indices = self.split_array == self.split_dict[split]
        self.X = self.filename_array[split_indices]
        self.y_array = torch.tensor(self.y_array[split_indices], dtype=torch.float)

        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            self.features_mat = torch.from_numpy(
                np.load(
                    os.path.join(
                        self.root_dir,
                        "features",
                        model_attributes[self.model_type]["feature_filename"],
                    ))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_coloredMNIST(
                self.model_type, train=True, augment_data=augment_data)
            self.eval_transform = get_transform_coloredMNIST(
                self.model_type, train=False, augment_data=augment_data)

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def get_num_classes(self):
        return torch.max(self.y_array).item() + 1

    def __len__(self):
        return self.y_array.shape[0]

    def __getitem__(self, idx):
        if model_attributes[self.model_type]["feature_type"] == "precomputed":
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(self.data_dir,
                                        self.filename_array[idx])
            img = Image.open(img_filename).convert("RGB")
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict[
                    "train"] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[idx]
                  in [self.split_dict["val"], self.split_dict["test"]]
                  and self.eval_transform):
                img = self.eval_transform(img)
            # Flatten if needed
            if model_attributes[self.model_type]["flatten"]:
                assert img.dim() == 3
                img = img.view(-1)
            x = img
        end_num_index = self.X[idx].index('.')
        img_idx = int(self.X[idx][4:end_num_index])
        if self.test_mnist:
            return {'image': x, 'label': self.y_array[idx], 'class_labels': self.class_labels[idx], 'idx': img_idx} # idx may be incorrect here
        else:
            return {'image': x, 'label': self.y_array[idx], 'idx': img_idx} # idx may be incorrect here

def get_transform_coloredMNIST(model_type, train, augment_data):
    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

def get_model_and_dataset_mnist(args, layer_num):#, target_name=3):
    dataset = ColoredMNISTDataset(
                root_dir=args.root_dir, 
                split='train', 
                target_name=args.target, 
                confounder_names=args.confounder_name, 
                model_type=args.model, 
                test_mnist=args.test_celebA
            )
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
    train_dataset = ColoredMNISTDataset(
                root_dir=args.root_dir, 
                split='train', 
                target_name=args.target, 
                confounder_names=args.confounder_name, 
                model_type=args.model, 
                test_mnist=args.test_celebA
            )
    val_dataset = ColoredMNISTDataset(
                root_dir=args.root_dir, 
                split='val', 
                target_name=args.target, 
                confounder_names=args.confounder_name, 
                model_type=args.model, 
                test_mnist=True
            )
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = create_dataloader(model, datasets, shared_dl_args, layer_num, class_labels=args.test_celebA)
    return dataloaders, model, dataset, shared_dl_args, num_classes

def collate_func(batch, pretrained_model, criterion, layer_num, return_class_labels=False):
    global tail_cache
    inputs = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    # print([item['class_labels'] for item in batch])exit()
    if return_class_labels:
        class_labels = torch.tensor([item['class_labels'] for item in batch])
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
    data = {'embeddings': embeddings, 'loss': loss, 'predicted_label': pred, 'actual_label': labels, 'idxs': idxs}
    if return_class_labels:
        data['class_label'] = class_labels
    tail_cache = dict()
    gc.collect()
    torch.cuda.empty_cache()

    return data

def create_dataloader(model, datasets, shared_dl_args, layer_num=1, criterion=nn.CrossEntropyLoss(reduction='none'), class_labels=False):
    if type(datasets) is dict:
        dataloaders = dict()
        for dataset_type in datasets:
                collate_fn = partial(collate_func, pretrained_model=model, criterion=criterion, layer_num=layer_num, return_class_labels=class_labels)
                dataloaders[dataset_type] = DataLoader(datasets[dataset_type], **shared_dl_args, collate_fn=collate_fn)
    else:
        # Create a DataLoader
        collate_fn = partial(collate_func, pretrained_model=model, criterion=criterion, layer_num=layer_num, return_class_labels=class_labels)
        dataloaders = DataLoader(datasets, **shared_dl_args, collate_fn=collate_fn)

    return dataloaders
    