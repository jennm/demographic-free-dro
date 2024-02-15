import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_classifiers_dataset import DROClassifiersDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.jigsaw_dataset import JigsawDataset
from classifier import LogisticRegressionModel

########################
####### SETTINGS #######
########################

confounder_settings = {
    "CelebA": {
        "constructor": CelebADataset
    },
    "CUB": {
        "constructor": CUBDataset
    },
    "MultiNLI": {
        "constructor": MultiNLIDataset
    },
    'jigsaw':{
        'constructor': JigsawDataset
    },
}


def read_classifiers(args):
    classifiers = list()
    models = torch.load(args.classifier_path)
    for i in range(int(args.num_classifiers)):
        model = LogisticRegressionModel(models['input_size'][i], models['num_classes'][i])
        model.load_state_dict(models['classifiers'][i])
        classifiers.append(model)
    return classifiers



########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    if args.dataset != "jigsaw":
        full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data,
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv"
        )
    else: 
        full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data,
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv",
            batch_size=args.batch_size
        )
    if args.classifier_groups:
        classifiers = read_classifiers(args)
        model = torch.load(args.model_path) # os.path.join(args.model_path, "last_model.pth"))
    else:
        classifiers = list()
        
    if return_full_dataset:
        if args.classifier_groups:
            return DROClassifiersDataset(
                full_dataset,
                process_item_fn=None,
                classifiers=classifiers,
                model=model,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
        else:
            return DRODataset(
                full_dataset,
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            )
    if train:
        splits = ["train", "val", "test"]
    else:
        splits = ["test"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    if args.classifier_groups:
        dro_subsets = [
            DROClassifiersDataset(
                subsets[split],
                process_item_fn=None,
                classifiers=classifiers,
                model=model,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            ) for split in splits
        ]
    else:
        dro_subsets = [
            DRODataset(
                subsets[split],
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str,
            ) for split in splits
        ]
    return dro_subsets
