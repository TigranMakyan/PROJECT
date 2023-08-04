import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms as T
from tqdm.auto import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import pyinstrument
import time
import sys
#import multiprocessing
#multiprocessing.set_start_method('spawn', True)


start_time = time.time()
sys.path.insert(0, '.')

from scalers.scale_light.blocks.stark_scale import build_stark_scale_model
from scalers.scale_light.blocks.stark_scale import build_stark_scale_model
from scalers.scale_light.utils import PreprocessorX, read_config
from data.dataset.light_dataset import create_light_dataloaders, Light_Dataset
#from data.dataset.split_dataset import ParentSampler
from utils import SaveBestModel, save_model, params_count_lite

profiler = pyinstrument.Profiler()
cfg_path = '/home/tigran/computer_vision/PROJECT/scalers/scale_light/configs/scale.yaml'
config = read_config(cfg_path)

DATA_PATH = config.DATA.PATH
BATCH_SIZE = config.TRAIN.BATCH_SIZE
LEARNING_RATE = config.TRAIN.LR
EPOCHS = config.TRAIN.EPOCH
device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.to(device)),
    T.Resize(size=(320, 320), antialias=True),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, hue=0.3, contrast=0.3),
    T.GaussianBlur(kernel_size=(5, 5), sigma=0.1),
    T.Normalize(mean=config.DATA.MEAN, std=config.DATA.STD)
])

transform_val = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.to(device)),
    T.Resize(size=(320, 320), antialias=True),
    T.Normalize(mean=config.DATA.MEAN, std=config.DATA.STD)
])

#def worker_init_fn(worker_id):
#    torch.manual_seed(worker_id)

transform = {'train': transform_train, 'val': transform_val}

train_dataset = Light_Dataset(config, transform_train, 'train')
valid_dataset = Light_Dataset(config, transform_val, 'val')

print('='*75)
#print(train_dataset[0])

#num_workers = multiprocessing.cpu_count()
#print(f'Num workers is {num_workers}')

train_loader, valid_loader = create_light_dataloaders(
    train_dataset, valid_dataset, batch_size=BATCH_SIZE
)
#SHOW ME THE COMPUTATION DEVICE
print(f'Computation device: {device}\n')

model = build_stark_scale_model(config, 'Train').to(device)
#model = build_relscaletransformer(pretrained=False).to(device)
#_initialize_weights(model)
#model = build_rel_scalenet(pretrained=False, freeze_bb=True, freeze=False).to(device)
print(next(model.parameters()).is_cuda)
print(model)
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load('/home/powerx/computer_vision/RelScale/models/model_adam_1_2206.pth')
#model.load_state_dict(checkpoint['model_state_dict'])
params_count_lite(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 25], gamma=0.3)
criterion = nn.MSELoss()
save_best_model = SaveBestModel()

def train_epoch(model, trainloader, optimizer, criterion, epoch):
    model.train()
    print('[INFO]: TRAINING IS RUNNING')
    train_running_loss = 0.0
    counter = 0
    shochik = -1
    with tqdm(enumerate(trainloader),total=len(trainloader)) as tepoch:
        for i, data in tepoch:
            tepoch.set_description(f"iter {i}/{len(trainloader)}")
            shochik+=1
            counter += 1
            images1, masks1, images2, masks2, scales = data
            if images1 is None or images2 is None:
                print('We passed a simple, because there is no image')
                continue
            optimizer.zero_grad()
            outputs = model(images1, images2, masks1, masks2)
            outputs = outputs.view(-1)
            loss = criterion(outputs, scales)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if counter % 50 == 0:
                tepoch.set_postfix(loss=train_running_loss/counter)
            if shochik % 100 == 0:
                print(outputs)
                with open('./logs/train_logs1.txt', 'a') as f:
                    f.write(f'Epoch: {epoch+1} Iter: {shochik/len(trainloader)*100:.0f}% Loss: \
                        {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
    
    epoch_loss = train_running_loss / counter
    return epoch_loss


def validate(model, testloader, criterion,epoch):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            images1, masks1, images2, masks2, scales = data
            outputs = model(images1, images2, masks1, masks2)
            outputs = outputs.view(-1)
            # calculate the loss
            loss = criterion(outputs, scales)
            valid_running_loss += loss.item()
            if counter % 5 == 0:
                print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
                with open('./logs/val_logs.txt', 'a') as f:
                    f.write(f'Epoch: {epoch} Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
            # calculate the accuracy
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    return epoch_loss

train_loss = []
valid_loss = []

#START THE TRAINING

for epoch in range(EPOCHS):
    print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
    train_epoch_loss = train_epoch(
        model=model,
        trainloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epoch=epoch
    )
    valid_epoch_loss = validate(model, valid_loader, criterion,epoch=epoch)
    scheduler.step()
    # scheduler.step(valid_epoch_loss)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(f'Training LOSS: {train_epoch_loss:.5f}')
    print(f'Validation LOSS: {valid_epoch_loss:.5f}')

    # save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_vgg.pth')
    save_model(epoch, model, optimizer, criterion, path_to_save=f'./weights/aug_model_{epoch}_0408.pth')
    print('=' * 75)

# #SAVE THE TRAINED MODEL WEIGHTS FOR FINAL TIME
# save_model(EPOCHS, model, optimizer, criterion, path_to_save='./models/final_model_tr.pth')
# print('TRAINING COMPLETE')


