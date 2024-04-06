import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset
from data.folds import Subset


class ConfounderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        model_type=None,
        augment_data=None,
    ):
        raise NotImplementedError

    # def update_LR_target(self, indices, new_values):
    #     if not hasattr(self, 'LR_targets'):
    #         self.LR_targets = []  
    #     self.LR_targets[indices] = new_values

    def get_group_array(self, use_classifier_groups=False):
        if use_classifier_groups: return self.classifier_group_array
        else: return self.group_array

    def get_n_groups(self, use_classifier_groups=False):
        if use_classifier_groups: return self.classifier_n_groups
        else: return self.n_groups

    def get_label_array(self):
        return self.y_array

    def get_up_weight_array(self):
        return self.up_weight_array

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.get_group_array()[idx]

        classifier_g = -1
        if hasattr(self, 'classifier_group_array'):
            classifier_group_array = self.get_group_array(use_classifier_groups=True)
            if idx < len(classifier_group_array):
                classifier_g = classifier_group_array[idx]

        up_weight = self.up_weight_array[idx]

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

        return x, y, g, up_weight, idx, classifier_g

    def get_splits(self, splits, train_frac=1.0, use_classifier_groups=False):
        subsets = {}
        for split in splits:
            assert split in ("train", "val",
                             "test"), f"{split} is not a valid split"
            mask = self.split_array == self.split_dict[split]

            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(
                    np.random.permutation(indices)[:num_to_retain])
            
            subsets[split] = Subset(self, indices, (use_classifier_groups and split != 'test'))
            # NOTE: the test set should NEVER use the classifier groups

        return subsets

    def group_str(self, group_idx, use_classifier_groups=False): # TODO: Check this
        if use_classifier_groups:
            group_name = f'{group_idx} / {self.get_n_groups(use_classifier_groups)}'
            return group_name

        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
            
        return group_name
