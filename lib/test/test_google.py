import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torchvision import transforms as T
from tqdm.auto import tqdm
import sys
sys.path.insert(0, '.')

from data.dataset.google_dataset import create_google_dataloaders, GoogleDataset_entire_image, create_google_datasets
from lib.train.utils import SaveBestModel, save_model, params_count_lite
from scalers.google.transformer_based import build_relscaletransformer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/media/storagedrive/map/', \
    help='input images folder dir')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size of dataloader')

args = vars(parser.parse_args())

DATA_PATH = args['data_path']
BATCH_SIZE = args['batch_size']
device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    T.Lambda(lambda x: x.to(device))
])

data_list = ['/media/storagedrive/scripts/test_data/']
pairs_list = ['/media/storagedrive/scripts/test_map_pairs.txt']

total_dataset = create_google_datasets(data_list, pairs_list, transform_test)
#CREATE DATASETS FOR TRAINING

#CREATE DATALOADERS FOR TRAINING
testloader = DataLoader(total_dataset, batch_size=BATCH_SIZE, shuffle=True)
#SHOW ME THE COMPUTATION DEVICE
print(f'Computation device: {device}\n')

model = build_relscaletransformer(pretrained=True).to(device)
print(model)
params_count_lite(model)

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
def criterion_percentile(y_pred, y_true):
    return torch.mean(100 * (torch.abs(y_pred - y_true) / y_true))

def test(model, testloader, criterion1, criterion2, criterion3):
    model.eval()
    print('Testign')
    valid_running_loss1, valid_running_loss2, valid_running_loss3 = 0.0, 0.0, 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            images1, images2, scales = data
            scales = torch.tensor(scales, device=torch.device(device), dtype=torch.float32)
           # scales = scales.to(device)
            # forward pass
            outputs = model(images1, images2)
            outputs = outputs.view(-1).to(torch.float32)
            # calculate the loss
            loss1 = criterion1(outputs, scales)
            loss2 = criterion2(outputs, scales)
            loss3 = criterion3(outputs, scales)
            valid_running_loss1 += loss1.item()
            valid_running_loss2 += loss2.item()
            valid_running_loss3 += loss3.item()
            if counter % 10 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
                with open('./logs/test_logs.txt', 'a') as f:
                    f.write(f'MSE Loss: {valid_running_loss1/counter:.8f} MAE Loss: {valid_running_loss1/counter:.8f}\n')
            # calculate the accuracy
        
    # loss and accuracy for the complete epoch
    epoch_loss1 = valid_running_loss1 / counter
    epoch_loss2 = valid_running_loss2 / counter
    epoch_loss3 = valid_running_loss3 / counter
    return epoch_loss1, epoch_loss2, epoch_loss3


l, l1, l2 = test(model, testloader, criterion_mse, criterion_l1, criterion_percentile)

print(f'RESULT of Testing: {l} MSE Loss')
print(f'RESULT of Testing: {l1} L1Loss')
print(f'RESULT of Testing: {l2} Percentile loss')
