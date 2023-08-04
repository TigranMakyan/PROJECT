from torchvision import transforms as T
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import os
import random
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_train = T.Compose([
    T.Lambda(lambda x: x.to(device)),
    T.ToTensor(),
    T.RandomGrayscale(0.1),
    T.RandomHorizontalFlip(), 
    T.ColorJitter(brightness=0.3, hue=0.3, contrast=0.3),
    T.GaussianBlur(kernel_size=(5, 5), sigma=0.1),
    T.Normalize(std=[0.225, 0.229, 0.224], mean=[0.485, 0.406, 0.456])
])

transform_val = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.to(device)),
    T.Normalize(std=[0.225, 0.229, 0.224], mean=[0.485, 0.406, 0.456])
])

transform = {'train': transform_train, 'val': transform_val}

class SplitDataset(Dataset):
    def __init__(self, indices, total_dataset, transform, cfg) -> None:
        super().__init__()
        self.dataset = [total_dataset[i] for i in indices]
        self.transform = transform
        self.cfg = cfg

    def __getitem__(self, index):
        item = self.dataset[index]
        image1_path = item[0]
        image2_path = item[1]
        scale = float(item[2])
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if self.transform is not None:
            image1_tensor = self.transform(image1)
            image2_tensor = self.transform(image2)

        mask1 = self._create_mask(image1_tensor.detach().cpu().permute(1, 2, 0).numpy(), mask_size=self.cfg.DATA.SEARCH.SIZE)
        mask2 = self._create_mask(image2_tensor.detach().cpu().permute(1, 2, 0).numpy(), mask_size=self.cfg.DATA.SEARCH.SIZE)

        return image1_tensor, mask1, image2_tensor, mask2, scale

    def __len__(self):
        return len(self.dataset)

    def _create_mask(self, img_arr: np.ndarray, mask_size: int):
        # Convert the image to grayscale
        print('Minchev grayscale gcely: ', img_arr.shape)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        mask_tensor = torch.from_numpy(edges).to(torch.bool).unsqueeze(dim=0)
        mask_tensor = mask_tensor.to('cuda')
        return mask_tensor


class ParentSampler:
    def __init__(self, cfg, transform: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        self.main_path = cfg.DATA.PATH
        self.data_path = os.path.join(self.main_path, 'data')
        self.pairs_path = os.path.join(self.main_path, 'pairs.txt')
        self.pairs_file = open(self.pairs_path, 'r')
        self.pairs = self.pairs_file.readlines()
        self.scale_dataset = []
        for item in self.pairs:
            item_info = item.split(' ')
            first_path = os.path.join(self.data_path, item_info[1])
            second_path = os.path.join(self.data_path, item_info[2])
            self.scale_dataset.append([first_path, second_path, item_info[3]])
        random.shuffle(self.scale_dataset)


    def train_val_datasets(self, ratio=0.1):
        train_idx, validation_idx = train_test_split(range(len(self.scale_dataset)),
                                             test_size=ratio,
                                             random_state=2023,
                                             shuffle=True)
    
        train_set = SplitDataset(train_idx, self.scale_dataset, self.transform['train'], self.cfg)
        val_set = SplitDataset(validation_idx, self.scale_dataset, self.transform['val'], self.cfg)
        return train_set, val_set 

