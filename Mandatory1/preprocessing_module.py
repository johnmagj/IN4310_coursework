import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import transforms

from PIL import Image


def find_class_names_filenames(img_dir):
    """
    Find class names based on folder names and list filenames class wise.
    
    Args:
        img_dir (str): root directory path, must folders of image files, each folder representing a class
    
    Return:
        class_names (str): list of class names, sorted
        class_filenames (list): list containing lists with filenames, each the indices of each sublist correspond to same index in class_names 
    """

    # List all class names (folder names) and sort the list
    class_names = os.listdir(img_dir)
    class_names.sort()

    # Create list containing lists of all filenames from each class
    # Each list containing filenames have same index as the index of class name in the sorted list class_names
    class_filenames = []
    for class_name in class_names:
        class_files_path = os.path.join(img_dir, class_name)
        class_filenames.append(os.listdir(class_files_path))

    return class_names, class_filenames

def get_path_from_dataset_indices(dataset, class_names, class_filenames, img_dir):
    """
    Helper function for stratified_split_data_paths().
    
    Args:
        dataset: list of [class_number, file_index] pairs
        class_names: list of class names
        class_filenames: list of lists, where each sub list contains filenames of class
        img_dir: root directory with class
    
    Return:
        list of filepaths
    """
    dataset_file_paths = []
    for class_idx, filename_idx in dataset:
        # print(class_idx, filename_idx, class_filenames[class_idx][filename_idx])

        class_name = class_names[class_idx]
        filename = class_filenames[class_idx][filename_idx]
        file_path = os.path.join(img_dir, class_name, filename)
        dataset_file_paths.append(file_path)

    return dataset_file_paths

def stratified_split_data_paths(img_dir, class_names, class_filenames):
    """
    1. Create array of indices for each file in each of the classes, in format: [[class_number, img_index], ... ]
    2. Use array of [class_number, img_index] pairs as x, and create a class_number target array, y 
    3. Split into into train, val and, test set
    4. Check if sets are disjoint
    
    Return: Lists of img paths and corresponding lists of class numbers
    """

    # Create one large array containing [class_number, filename_index] pairs of all the images based on the ordering in class_filenames
    class_filenames_indices = []
    for class_number, class_filenames_list in enumerate(class_filenames):
        idx_class_filenames = list(range(len(class_filenames_list)))

        for idx in idx_class_filenames:
            class_filenames_indices.append([class_number, idx])

    class_filenames_indices = np.asarray(class_filenames_indices, dtype=np.int32)

    # Make array of target (class number) y to correspond to list of [class_number, file_index] pairs
    x = class_filenames_indices
    y = class_filenames_indices[:,0]

    # Two step process of spliting the indices into train-, val- and test-set, about 70, 10 and 20% respectively
    x_train_indices, x_valtest_indices, y_train, y_valtest = train_test_split(x, y, train_size=0.70, stratify=y)
    x_val_indices, x_test_indices, y_val, y_test= train_test_split(x_valtest_indices, 
                                                                                    y_valtest, 
                                                                                    train_size=0.40, 
                                                                                    test_size=0.60, 
                                                                                    stratify=y_valtest)

    # Get the paths from the indices
    x_train_paths = get_path_from_dataset_indices(x_train_indices, class_names, class_filenames, img_dir)
    x_val_paths = get_path_from_dataset_indices(x_val_indices, class_names, class_filenames, img_dir)
    x_test_paths = get_path_from_dataset_indices(x_test_indices, class_names, class_filenames, img_dir)

    # Check if the sets are disjoint
    not_disjoint = 0
    if not set(x_train_paths).isdisjoint(set(x_val_paths)):
        print("Train and val set not disjoint")
        not_disjoint += 0

    if not set(x_train_paths).isdisjoint(set(x_test_paths)):
        print("Train and test set not disjoint")
        not_disjoint += 0

    if not set(x_val_paths).isdisjoint(set(x_test_paths)):
        print("Val and test set not disjoint")
        not_disjoint += 0

    if not_disjoint > 0:
        raise ValueError("Datasets not disjoint")


    return x_train_paths, x_val_paths, x_test_paths, y_train, y_val, y_test

# Class which turn file paths into pipeline-friendly object  
# Inherit abstract class: torch.utils.data.Dataset, this to make it compatible with the pytorch pipeline
# Need __init__, __len__ and __getitem__
# Inspired by 02_CNN_Example.ipynb

class NatureCityScenesDataset(torch.utils.data.Dataset):
    def __init__(self, x_paths, y_class_number, transform=None):

        self.image_paths = x_paths
        self.image_label = y_class_number

        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Convert to RGB (if some images are RGBA or Grayscale)
        image = Image.open(img_path).convert("RGB")

        label = int(self.image_label[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label