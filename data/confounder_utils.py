import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from data.celebA_dataset import CelebADataset
from data.colored_mnist_dataset import ColoredMNISTDataset
from data.colored_mnist_hard_dataset import ColoredMNIST_HARD_Dataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.jigsaw_dataset import JigsawDataset

from functools import partial

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
    "ColoredMNIST" : {
        'constructor': ColoredMNISTDataset
    },
    "ColoredMNIST_HARD" : {
        'constructor': ColoredMNIST_HARD_Dataset
    }
}


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
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv",
            classifier_group_path=args.classifier_group_path, # all this does is optionally load the classifier group arrays as a separate dataset member
        )
    else: 
        full_dataset = confounder_settings[args.dataset]["constructor"](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data,
            metadata_csv_name=args.metadata_csv_name if (args.metadata_csv_name is not None) else "metadata.csv",
            batch_size=args.batch_size,
            classifier_group_path=args.classifier_group_path,
        )
        
    # DRODataset( ConfounderDataset )
    if return_full_dataset: # get_group_array here calls confounder_dataset's get_group_array directly
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.get_n_groups(args.classifier_group_path != ''),
            n_classes=full_dataset.n_classes,
            group_str_fn=partial(full_dataset.group_str, use_classifier_groups=(args.classifier_group_path != '')),
            use_classifier_groups=(args.classifier_group_path != '')
        )

    if train:
        splits = ["train", "val", "test"]
    else:
        splits = ["test"]

    # Subset( ConfounderDataset )
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction, use_classifier_groups=(args.classifier_group_path != ''))
    # we're telling subsets to use classifier groups or regular groups, so we know the get_group_array here will do the right thing

    # DRODataset( Subset ( ConfounderDataset ) )
    dro_subsets = [ # the call to get_group_array here goes to subset's get_group_array goes to confounder_dataset's get_group_array
         DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.get_n_groups((args.classifier_group_path != '' and split == 'train')),
            n_classes=full_dataset.n_classes,
            group_str_fn=partial(full_dataset.group_str, use_classifier_groups=(args.classifier_group_path != '' and split == 'train')),
            use_classifier_groups=(args.classifier_group_path != '' and split == 'train')
        ) for split in splits
    ] # NOTE: changed so only train uses classifier groups

    return dro_subsets