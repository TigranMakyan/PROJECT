import torch
from tqdm.auto import tqdm

class LightTrainer:
    def __init__(self, model, loaders, optimizer, settings, criterion, lr_scheduler=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = loaders
        self.settings = settings
        self.lr_scheduler = lr_scheduler
        self.move_data_to_gpu = getattr(self.settings, 'move_data_to_gpu', True)
        self.criterion = criterion

    def train_epoch(self, epoch):
        self.model.train()
        print('[INFO]: TRAINING IS RUNNING')
        train_running_loss = 0.0
        counter = 0
        shochik = -1
        with tqdm(enumerate(self.train_loader),total=len(self.train_loader)) as tepoch:
            for i, data in tepoch:
                tepoch.set_description(f"iter {i}/{len(self.train_loader)}")
                shochik+=1
                counter += 1
                images1, masks1, images2, masks2, scales = data
                if images1 is None or images2 is None:
                    print('We passed a simple, because there is no image')
                    continue
                self.optimizer.zero_grad()
                outputs = self.model(images1, images2, masks1, masks2)
                outputs = outputs.view(-1)
                loss = self.criterion(outputs, scales)
                train_running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if counter % 50 == 0:
                    tepoch.set_postfix(loss=train_running_loss/counter)
                if shochik % 100 == 0:
                    print(outputs)
                    with open('./logs/train_logs1.txt', 'a') as f:
                        f.write(f'Epoch: {epoch+1} Iter: {shochik/len(self.train_loader)*100:.0f}% Loss: \
                            {train_running_loss/counter:.8f} Batch_Loss: {loss.item():.8f}\n')
        
        epoch_loss = train_running_loss / counter
        return epoch_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        print('Validation')
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(self.train_loader, total=len(self.train_loader)):
                counter +=1
                images1, masks1, images2, masks2, scales = data
                outputs = self.model(images1, images2, masks1, masks2)
                outputs = outputs.view(-1)
                loss = self.criterion(scales, outputs)
                valid_running_loss += loss.item()
                if counter % 5 == 0:
                    print(f'outputs|labels\n {torch.stack((outputs,scales),1)}')
                    with open('./logs/val_logs.txt', 'a') as f:
                        f.write(f'Epoch: {epoch} Loss: {valid_running_loss/counter:.8f} Batch_Loss: {loss.item()}\n')
        epoch_loss = valid_running_loss / counter
        return epoch_loss



    def train(self):
        train_loss, valid_loss = [], []
        EPOCHS = getattr(self.settings, 'epochs', 50)
        for epoch in range(EPOCHS):
            print(f'[INFO]: Epoch {epoch+1} of {EPOCHS}')
            train_epoch_loss = self.train_epoch(epoch)
            valid_epoch_loss = self.validate_epoch(epoch)
            self.lr_scheduler.step()
            # scheduler.step(valid_epoch_loss)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            print(f'Training LOSS: {train_epoch_loss:.5f}')
            print(f'Validation LOSS: {valid_epoch_loss:.5f}')

            # save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, data_path='./weights/model_vgg.pth')
            self.save_model()
            print('=' * 75)
