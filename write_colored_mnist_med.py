import os
import pandas as pd
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

"test_shift == random, cmap=hsv"
class ColoredMNIST(Dataset):

    def __init__(self, data, targets, train=True, transform=None, seed=None):
        self.seed = seed
        self.p_correlation = 0.90
        self.cmap = 'hsv'

        self.class_map = {0: 0, 1: 0, 
                          2: 1, 3: 1, 
                          4: 2, 5: 2, 
                          6: 3, 7: 3, 
                          8: 4, 9: 4}
        
        self.classes = list(self.class_map.keys()) # list of original classes i.e. num digits in MNIST: 0 1 2 3 4 5 6 7 8 9
        self.new_classes = np.unique(list(self.class_map.values())) # list of new classes: 0 1 2 3 4

        self.test_classes = [x for x in np.unique(targets) if x not in self.classes] # accounts for any digits not in in train set

        self.p_correlation = [self.p_correlation] * len(self.new_classes) # sets correlation rate for each of the new classes

        self.train = train
        self.test_shift = "random"
        self.transform = transform

        # filter for train_classes: for each data point, counts how many classes i'm in
        class_filter = torch.stack([(targets == i) for i in self.classes]).sum(dim=0)
        # selects target labels that actually occur
        self.targets = targets[class_filter > 0]
        # selects data points that are in at least 1 class
        data = data[class_filter > 0]

        self.targets_all = {'spurious': np.zeros(len(self.targets), dtype=int)}
        
        # update targets with new class ids
        self.targets = torch.tensor([self.class_map[t.item()] for t in self.targets],
                                    dtype=self.targets.dtype)
        # stores the new target tensor in targets_all
        self.targets_all['target'] = self.targets.numpy()
        
        # Colors + Data
        self.colors = self._init_colors(self.cmap)
        if data.shape[1] != 3:   # Add RGB channels
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
        self.data = self._init_data(data)
     
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        color_id = self.targets_all['spurious'][idx]
        return (sample, self.targets[idx], f'C{color_id}')

    def _init_colors(self, cmap):
        # Initialize list of RGB color values
        try:
            cmap = cm.get_cmap(cmap)
        except ValueError:  # single color
            cmap = self._get_single_color_cmap(cmap)
        cmap_vals = np.arange(0, 1, step=1 / (2 * len(self.new_classes)))
        colors = []
        # for ix, _ in enumerate(self.new_classes):
        for ix in range(2 * len(self.new_classes)):
            rgb = cmap(cmap_vals[ix])[:3]
            rgb = [int(float(x)) for x in np.array(rgb) * 255]
            colors.append(rgb)
        return colors

    def _get_single_color_cmap(self, c):
        rgb = to_rgb(c)
        r1, g1, b1 = rgb
        cdict = {'red':   ((0, r1, r1),
                           (1, r1, r1)),
                 'green': ((0, g1, g1),
                           (1, g1, g1)),
                 'blue':  ((0, b1, b1),
                           (1, b1, b1))}
        cmap = LinearSegmentedColormap('custom_cmap', cdict)
        return cmap

    def _init_data(self, data):
        np.random.seed(self.seed)
        pbar = tqdm(total=len(self.targets), desc='Initializing data')

        # for each new class
        for ix, c in enumerate(self.new_classes):
            # get the idx where this new class occurs
            class_ix = np.where(self.targets == c)[0]

            # for each item in this new class, binomial dist w p_corr to determine whether to spuriously color or not
            is_spurious = np.random.binomial(1, self.p_correlation[ix],
                                             size=len(class_ix))
            
            # for each item in the new class
            for cix_, cix in enumerate(class_ix):
                pixels_r = np.where(
                    np.logical_and(data[cix, 0, :, :] >= 120,
                                   data[cix, 0, :, :] <= 255))
    
                color_ix = (ix if is_spurious[cix_] else ix + 5) # majority colors [0, 1, 2, 3, 4], minority colors [5, 6, 7, 8, 9]
                color = self.colors[color_ix]
                data[cix, :, pixels_r[0], pixels_r[1]] = (
                    torch.tensor(color, dtype=torch.uint8).unsqueeze(1).repeat(1, len(pixels_r[0])))
                
                self.targets_all['spurious'][cix] = int(color_ix)
                pbar.update(1)

        return data.float() / 255  # For normalization


def train_val_split(dataset, val_split, seed):
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def load_colored_mnist(root_dir, train_shuffle=True, transform=None):
    print('load colored mnist')
    data_folder = os.path.join(root_dir, "coloredMNIST_HARD")
    mnist_folder = os.path.join(data_folder, "data", "mnist_imgs")
    colored_mnist_folder = os.path.join(data_folder, "data", "colored_mnist_imgs")
    metadata_csv_path = os.path.join(data_folder, "data", "metadata.csv")
    print('finished setting up data paths')

    val_split = 0.1
    seed = 42

    print('starting mnist download')
    mnist_train = torchvision.datasets.MNIST(root=mnist_folder, 
                                             train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=mnist_folder, 
                                            train=False, download=True)
    print('finished downloading mnist')

    transform = (transforms.Compose([transforms.Resize(40),
                                     transforms.RandomCrop(32, padding=0),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))])
                 if transform is None else transform)
    
    # Split original train set into train and val
    train_indices, val_indices = train_val_split(mnist_train, 
                                                 val_split,
                                                 seed)
    train_data = mnist_train.data[train_indices]
    train_targets = mnist_train.targets[train_indices]
    val_data = mnist_train.data[val_indices]
    val_targets = mnist_train.targets[val_indices]
    print('finished splitting datasets')
    
    print('starting to color dataset')
    colored_mnist_train = ColoredMNIST(data=train_data,
                                       targets=train_targets,
                                       train=True, transform=transform, seed=seed)
    
    # Val set is setup with same data distribution as test set by convention.
    colored_mnist_val = ColoredMNIST(data=val_data, targets=val_targets,
                                         train=True, transform=transform, seed=seed)
        

    colored_mnist_test = ColoredMNIST(data=mnist_test.data,
                                      targets=mnist_test.targets,
                                      train=True, transform=transform, seed=seed)

    print('finished coloring datasets')
    metadata = []

    # write files
    print('starting to write files')
    idx = 0
    for (img, digit_class, color_class) in colored_mnist_train:
        image_id = f'img_{idx}.png'
        image_pil = TF.to_pil_image(img)
        image_pil.save(os.path.join(colored_mnist_folder, image_id))
        metadata.append({'split': 0, 'image_id': image_id, 
                         "0": int(digit_class == 0), 
                         "target": int(digit_class == 1), 
                         "2": int(digit_class == 2), 
                         "3": int(digit_class == 3), 
                         "4": int(digit_class == 4), 
                         "C0": int(color_class == "C0"),
                         "C1": int(color_class == "C1"),
                         "C2": int(color_class == "C2"),
                         "C3": int(color_class == "C3"),
                         "C4": int(color_class == "C4"),
                         "C5": int(color_class == 'C5'),
                         "confounder": int(color_class == "C6"),
                         "C7": int(color_class == "C7"),
                         "C8": int(color_class == "C8"),
                         "C9": int(color_class == "C9")
                         }
                    )
        idx += 1

    for (img, digit_class, color_class) in colored_mnist_val:
        image_id = f'img_{idx}.png'
        image_pil = TF.to_pil_image(img)
        image_pil.save(os.path.join(colored_mnist_folder, image_id))
        metadata.append({'split': 1, 'image_id': image_id, 
                         "0": int(digit_class == 0), 
                         "target": int(digit_class == 1), 
                         "2": int(digit_class == 2), 
                         "3": int(digit_class == 3), 
                         "4": int(digit_class == 4), 
                         "C0": int(color_class == "C0"),
                         "C1": int(color_class == "C1"),
                         "C2": int(color_class == "C2"),
                         "C3": int(color_class == "C3"),
                         "C4": int(color_class == "C4"),
                         "C5": int(color_class == 'C5'),
                         "confounder": int(color_class == "C6"),
                         "C7": int(color_class == "C7"),
                         "C8": int(color_class == "C8"),
                         "C9": int(color_class == "C9")
                         }
                    )
        idx += 1

    for (img, digit_class, color_class) in colored_mnist_test:
        image_id = f'img_{idx}.png'
        image_pil = TF.to_pil_image(img)
        image_pil.save(os.path.join(colored_mnist_folder, image_id))
        metadata.append({'split': 2, 'image_id': image_id, 
                         "0": int(digit_class == 0), 
                         "target": int(digit_class == 1), 
                         "2": int(digit_class == 2), 
                         "3": int(digit_class == 3), 
                         "4": int(digit_class == 4), 
                         "C0": int(color_class == "C0"),
                         "C1": int(color_class == "C1"),
                         "C2": int(color_class == "C2"),
                         "C3": int(color_class == "C3"),
                         "C4": int(color_class == "C4"),
                         "C5": int(color_class == 'C5'),
                         "confounder": int(color_class == "C6"),
                         "C7": int(color_class == "C7"),
                         "C8": int(color_class == "C8"),
                         "C9": int(color_class == "C9")
                         }
                    )
        idx += 1

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(metadata_csv_path, index=False)

    print('finished writing files')

load_colored_mnist('./')