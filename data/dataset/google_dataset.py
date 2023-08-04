import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random

import sys
sys.path.insert(0, '.')
from data.utils.image_utils import image_processing

NUM_WORKERS=0

class GoogleDataset(Dataset):

    def __init__(self, data_path='/home/user/Datasets/map/', transform=None) -> None:
        super().__init__()
        print(f'BUILDING GOOGLE DATASET')
        self.transform = transform
        self.data_path = data_path
        self.pairs_path = os.path.join(data_path, 'pairs_sampled.txt')
        
        self.pairs_file = open(self.pairs_path, 'r')
        self.pairs = self.pairs_file.readlines()

        self.scale_dataset = []
        for item in self.pairs:
            item_info = item.split(' ')
            self.scale_dataset.append([item_info[0], item_info[1], item_info[-1]])
        random.shuffle(self.scale_dataset)


    def __len__(self):
        return len(self.scale_dataset)

    def __getitem__(self, index):
        item = self.scale_dataset[index]
        image1 = cv2.imread(os.path.join(self.data_path, item[0]))
        image2 = cv2.imread(os.path.join(self.data_path, item[1]))
        scale = item[-1]

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, scale

class GoogleMapDataset(Dataset):
    
    def __init__(self, data_path, pairs_path, transform=None):
        super().__init__()
        print('BUILDING GOOGLE DATASET')
        self.transform = transform
        self.data_path = data_path
        self.pairs_path = pairs_path

        self.pairs_file = open(self.pairs_path, 'r')
        self.pairs = self.pairs_file.readlines()
        self.scale_dataset = []
        for item in self.pairs:
            item_info = item.split(' ')
            self.scale_dataset.append([item_info[0], item_info[1], item_info[2]])
        random.shuffle(self.scale_dataset)

    def __len__(self):
        return len(self.scale_dataset)

    def __getitem__(self, index):
        item = self.scale_dataset[index]
        image1_path = item[0]
        image2_path = item[1]
        scale =float(item[2])
        
        

        image1, image2 = image_processing(img1_path=image1_path, img2_path=image2_path, image_transforms=self.transform, device='cuda')
        return image1, image2, scale

class GoogleDataset_entire_image(Dataset):

    def __init__(self, data_path='/home/user/Datasets/map/', transform=None) -> None:
        super().__init__()
        print(f'BUILDING GOOGLE DATASET')
        self.transform = transform
        self.data_path = data_path
        self.pairs_path = os.path.join(data_path, 'pairs_sampled.txt')
        
        self.pairs_file = open(self.pairs_path, 'r')
        self.pairs = self.pairs_file.readlines()

        self.scale_dataset = []
        for item in self.pairs:
            item_info = item.split(' ')
            self.scale_dataset.append([item_info[0], item_info[1], item_info[-1]])
        random.shuffle(self.scale_dataset)


    def __len__(self):
        return len(self.scale_dataset)

    def __getitem__(self, index):
        item = self.scale_dataset[index]
        image1_path = os.path.join(self.data_path, item[0])
        image2_path = os.path.join(self.data_path, item[1])
        scale = item[-1]
        
        image1, image2 = image_processing(img1_path=image1_path,
                                        img2_path=image2_path, 
                                        image_transforms=self.transform, 
                                        device='cuda')
        
        return image1, image2, scale


def create_google_dataloaders(train_set, val_set, batch_size):
    '''
    Building dataloaders for training
    '''
    train_datalaoder = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_datalaoder = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    
    print('DATALOADERS ARE READY')
    return train_datalaoder, val_datalaoder

def create_google_datasets(data_paths_list, pairs_paths_list, transform=None):
    train_sets_list = [GoogleMapDataset(data_path=d, pairs_path=p, transform=transform) for d, p in zip(data_paths_list, pairs_paths_list)]
    data = ConcatDataset(train_sets_list)
    print('Total training dataset is READY')
    return data
