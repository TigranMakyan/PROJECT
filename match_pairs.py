import torch
import torch.nn as nn
import argparse
from torchvision import transforms as T
import cv2

from data.image_utils import image_processing
from models.transformer_based import build_relscaletransformer

parser = argparse.ArgumentParser()
parser.add_argument('-im1', '--image1_path', type=str, default='', help='image 1 path')
parser.add_argument('-im2', '--image2_path', type=str, default='', help='image 2 path')
parser.add_argument('-device', type=str, default='cpu')

args = vars(parser.parse_args())
device = torch.device(args['device'])
image1_path = args['image1_path']
image2_path = args['image2_path']

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4604, 0.4661, 0.4107], std=[0.1967, 0.1825, 0.1944]), 
    # T.Lambda(lambda x: x.to(device))
])

image1, image2 = image_processing(image1_path, image2_path, transform_test, device=device)
image1 = torch.unsqueeze(image1, dim=0)
image2 = torch.unsqueeze(image2, dim=0)
model = build_relscaletransformer(pretrained=True)

with torch.no_grad():
    result = model(image1, image2)

print("RELATIVE SCALE IS: ", result)

