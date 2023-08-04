import torch
import torch.nn as nn
from torchvision import transforms as T
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from scalers.scale_light.blocks.stark_scale import build_stark_scale_model
from scalers.scale_light.utils import PreprocessorX, read_config
from data.dataset.light_dataset import  create_light_datasets
from lib.train.utils import SaveBestModel, params_count_lite, percentile_loss

cfg_path = '/home/tigran/computer_vision/PROJECT/scalers/scale_light/configs/scale.yaml'
config = read_config(cfg_path)

DATA_PATH = config.DATA.VAL.PATH
BATCH_SIZE = config.TRAIN.BATCH_SIZE
device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    T.Lambda(lambda x: x.to(device))
])

data_list = config.DATA.VAL.LIST
pairs_list = config.DATA.VAL.PAIRS_LIST
preprocessor = PreprocessorX()

total_dataset = create_light_datasets(DATA_PATH, data_list, pairs_list, preprocessor)
#CREATE DATASETS FOR TRAINING
asenq_nkar, _, _, _, _ = total_dataset[0]
#print(f'NAYI APER: {asenq_nkar.shape}')
total_dataloader = DataLoader(total_dataset, batch_size=BATCH_SIZE, shuffle=True)
#SHOW ME THE COMPUTATION DEVICE
print(f'Computation device: {device}\n')

model = build_stark_scale_model(config, 'Train').to(device)
#model = build_relscaletransformer(pretrained=False).to(device)
#_initialize_weights(model)
#model = build_rel_scalenet(pretrained=False, freeze_bb=True, freeze=False).to(device)
#print(model)
#model = torch.nn.DataParallel(model)
checkpoint = torch.load('/home/tigran/computer_vision/PROJECT/weights/light_model_7_1207.pth')
model.load_state_dict(checkpoint['model_state_dict'])
params_count_lite(model)

criterion = nn.MSELoss()
#percentile = percentile_loss
l1_loss = nn.L1Loss()
#save_best_model = SaveBestModel()

@torch.no_grad()
def testing(model, testloader, criterion, l1_criterion):
    model.eval()
    print('Testing')
    valid_running_loss, valid_running_l1, valid_running_percent = 0.0, 0.0, 0.0
    loss_list = []
    l1_list = []
    max_list = []
    percent_list = []
    counter = 0
    for i, data in tqdm(enumerate(testloader), total=len(testloader)):
        counter += 1
        
        images1, masks1, images2, masks2, scales = data
        images1 = images1.squeeze().to(device)
        images2 = images2.squeeze().to(device)
        masks1 = masks1.squeeze().to(device)
        masks2 = masks2.squeeze().to(device)
        scales = torch.tensor(scales, device=torch.device(device), dtype=torch.float32)
        # scales = scales.to(device)
        # forward pass
        outputs = model(images1, images2, masks1, masks2)
        outputs = outputs.view(-1).to(torch.float32)
        # calculate the loss
        print(f'outputs: {outputs.shape}, scales: {scales.shape}')
       
        loss = criterion(outputs, scales)
        loss_list.append(loss)
        l1_loss = l1_criterion(outputs, scales)
        asenq = l1_loss / scales
        percent_list.append((torch.mean(asenq) * 100))
        l1_list.append(l1_loss)
        #percentile_value = percentile(outputs, scales)
        #print(loss)
        valid_running_loss += loss.item()
        valid_running_l1 += l1_loss.item()
        #valid_running_percent += percentile_value.item()

        if counter % 5 == 0:
            print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
            with open('./logs/test_logs.txt', 'a') as f:
                f.write(f'Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
        # calculate the accuracy
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    #epoch_loss_p = valid_running_percent / counter
    epoch_loss_l1 = valid_running_l1 / counter
    loss_list = [i.item() for i in loss_list]
    l1_list = [j.item() for j in l1_list]

    
    print(f'Maximal MSE loss value is {max(loss_list):.5f}')
    print(f'Mean MSE loss value is {(sum(loss_list) / len(loss_list)):.5f}')
    print(f'Maximal L1 loss value is {max(l1_list):.5f}')
    print(f'Mean L1 loss value is {(sum(l1_list) / len(l1_list))}:.5f')
    print(f'Maximal percentile loss value is {max(percent_list)}:.5f ')
    print(f'Mean percentile loss value is {(sum(percent_list) / len(percent_list))} ')
    return epoch_loss, epoch_loss_l1 #, epoch_loss_p, epoch_loss_l1

    
valid_loss, l1_result = testing(model, total_dataloader, criterion, l1_loss)

print(f'Validation LOSS: {valid_loss:.5f}\n')
#print(f'Validation LOSS: {valid_percent:.5f}\n')
print(f'Validation LOSS: {l1_result:.5f}\n')


