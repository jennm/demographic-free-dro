import os
import pandas as pd
from PIL import Image
import numpy as np
from models import model_attributes
import torchvision.transforms as transforms
from data.confounder_dataset import ConfounderDataset
import torch


'''
metadata.csv
Cols: image_id, the 5 digit classes, the 5 colors, split
need actual imgs to exist in root/data/colored_mnist_imgs
'''

class ColoredMNIST_HARD_Dataset(ConfounderDataset):
    def __init__(self,
                 root_dir,
                 target_name,
                 confounder_names,
                 model_type,
                 augment_data,
                 metadata_csv_name="metadata.csv",
                 classifier_group_path=''):
        
        self.root_dir = os.path.join(root_dir, "coloredMNIST_HARD")
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type

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

        target_idx = self.attr_idx(self.target_name) 
        self.y_array = self.attrs_df[:, target_idx]
        self.up_weight_array = torch.ones(len(self.y_array))
        self.n_classes = 2 

        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        self.confounder_array = np.matmul(
            confounders.astype(int),
            np.power(2, np.arange(len(self.confounder_idx)))
        )

        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array * (self.n_groups / 2)
                            + self.confounder_array).astype("int")

        if classifier_group_path:
            # npzfile = np.load('classifier_groups.npz')
            # group_info = npzfile['group_array']
            # self.classifier_group_array = group_info
            # self.classifier_n_groups = self.classifier_group_array.shape[1]

            aug_df = pd.read_csv('results/ColoredMNIST_HARD/ColoredMNIST_HARD_TEST/train_downstream_ERM_upweight_0_epochs_5_lr_0.001_weight_decay_0.0001/final_epoch1/metadata_aug.csv')
            self.misclassified = aug_df['wrong_1_times'].values
            print(self.misclassified.shape)

            self.classifier_n_groups = 4
            self.classifier_group_array = np.stack(
                [
                np.array([  0, 
                            1 if self.y_array[i] else -1, 
                            2 if self.confounder_array[i] else -1,
                            3 if i < 54000 and self.misclassified[i] else -1
                        ]) 
                for i in range(len(self.y_array))
                ]
            )

            self.classifier_group_array = torch.tensor(self.classifier_group_array)


        self.split_df = pd.read_csv(
            os.path.join(self.root_dir, "data", metadata_csv_name)
        )
        self.split_array = self.split_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2
        }

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
    
    def update_group_array(self, group_array):
        self.group_array = group_array

    def update_up_weight_array(self, new_up_weight_array):
        self.up_weight_array = new_up_weight_array
    
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
