import yaml
from easydict import EasyDict
import cv2
import numpy as np
import torch

def read_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = EasyDict(config_dict)
    return config

def params_count_lite(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {total_trainable_params}')

def create_mask(image, mask_size: tuple):    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.resize(edges, mask_size)
    return edges

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def get_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]
    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask

class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view((1, 3, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view((1, 3, 1, 1))

    def create_mask(self, img_arr: np.ndarray, mask_size: int):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        return edges

    def process(self, img_arr: np.ndarray, mask_size: int):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Apply a blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.resize(edges, (mask_size, mask_size))
        img_arr = cv2.resize(img_arr, (mask_size, mask_size))
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr, device='cuda').float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(edges).to('cuda').to(torch.bool).unsqueeze(dim=0)  # (1,H,W)
        # print('In process ing: ', img_tensor_norm.shape)
        # print('In process mask: ', amask_tensor.shape)
        return img_tensor_norm, amask_tensor