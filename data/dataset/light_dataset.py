import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import jpeg4py
from kornia.filters import canny

class Light_Dataset(Dataset):
    '''
    From config file we get:
    - DATA.PATH -> /home/tigran/scripts
    - DATA.MEAN ->
    - DATA.STD ->
    '''
    def __init__(self, cfg, transform: object, mode: str ='train') -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['train', 'val']
        self.transform = transform
        self.main_path = cfg.DATA.PATH
        self.data_path = os.path.join(self.main_path, 'data')
        self.scale_dataset = self._collect_data()

    def _collect_data(self):
        if self.mode == 'train':
            pairs_path = os.path.join(self.main_path, 'pairs_train.txt')
        else:
            pairs_path = os.path.join(self.main_path, 'pairs_val.txt')

        with open(pairs_path, 'r') as f:
            pairs = f.readlines()
        scale_dataset = []
        for item in pairs:
            item_info = item.split(' ')
            first_path = os.path.join(self.data_path, item_info[1])
            second_path = os.path.join(self.data_path, item_info[2])
            scale_dataset.append([first_path, second_path, item_info[3]])

        random.shuffle(scale_dataset)
        return scale_dataset

    def __getitem__(self, index):
        item = self.scale_dataset[index]
        image1_path = item[0]
        image2_path = item[1]
        scale = torch.tensor(float(item[2]), device='cuda')
        image1 = self.image_loader(image1_path)
        image2 = self.image_loader(image2_path)

        if self.transform is not None:
            image1_tensor = self.transform(image1)
            image2_tensor = self.transform(image2)
        
        #mask1 = self._create_mask(image1_tensor.detach().cpu().permute(1, 2, 0).numpy(), mask_size=self.cfg.DATA.SEARCH.SIZE)
        #mask2 = self._create_mask(image2_tensor.detach().cpu().permute(1, 2, 0).numpy(), mask_size=self.cfg.DATA.SEARCH.SIZE)
        
        mask1 = self._create_mask_tensor(image1_tensor)
        mask2 = self._create_mask_tensor(image2_tensor)
        #print(f'In Dataset getitem mask is created with shape" {mask1.shape}, image shape: {image1_tensor.shape}')

        return image1_tensor, mask1, image2_tensor, mask2, scale
    
    def __len__(self):
        return len(self.scale_dataset)
    
    def _create_mask(self, img_arr: np.ndarray, mask_size: int):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #print(blurred.shape, type(blurred))
        # Perform edge detection
        blurredCopy = np.uint8(blurred)
        edges = cv2.Canny(blurredCopy, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        mask_tensor = torch.from_numpy(edges).to(torch.bool).unsqueeze(dim=0)
        mask_tensor = mask_tensor.to('cuda')
        return mask_tensor

    def _create_mask_tensor(self, img_arr: torch.Tensor):
        img_copy = torch.clone(img_arr).unsqueeze(0)
        _, edges = canny(img_copy)
        return edges.squeeze()

    def image_loader(self, path):
        try: 
            return jpeg4py.JPEG(path).decode()
        except Exception as e:
            print(f'ERROR: Could not read image {path}')
            print(e)
            return None

def create_light_dataloaders(train_set, val_set, batch_size):
    '''
    Building dataloaders for training
    '''
    train_datalaoder = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_datalaoder = DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )
    
    print('DATALOADERS ARE READY')
    return train_datalaoder, val_datalaoder

