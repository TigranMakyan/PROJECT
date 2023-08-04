import torch.nn as nn
import torch
import os
import shutil
import cv2
from collections import OrderedDict
import copy

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def percentile_loss(y_pred, y_true):
    return (y_pred - y_true) / y_true

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, data_path='weights/best_model.pth'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, data_path)


def save_model(epochs, model, optimizer, criterion, path_to_save='weights/model_vgg_just.pth'):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path_to_save)


def handle_error_with_dilation():
    """
    Convolution layers, which are used with dilation can lead to
    "slow_conv_dilated<>" not implemented for 'Byte' error, 
    so for handling it you can use this function, because the reason of
    error is dtype of your argument
    """
    dtypes = (torch.uint8, torch.int8, torch.int16,
          torch.int32, torch.int64, torch.float,
          torch.double)

    for dtype in dtypes:
        print(dtype)
        t = torch.randn(1, 4, 5, 5).to(dtype)
        w = torch.randn(3, 4, 3, 3).to(dtype)

        torch.nn.functional.conv2d(t, w)
        try:
            torch.nn.functional.conv2d(t, w, dilation=(2, 2))
        except Exception as e:
            print(dtype, ":", e)


def data_moving(source_folder, destination_train, destination_test):
    """
    This function will help you for splitting your imagery and creating train and test 
    data folders in your preferable path
    """

    i = 0
    for file_name in os.listdir(source_folder):
        if i % 20 == 0:
            # construct full file path
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_test, file_name)
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
                print('Moved:', file_name)
        else:
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_train, file_name)
            if os.path.isfile(source):
                shutil.move(source, destination)
                print('Moved:', file_name)
        i += 1

def move_images_to_their_folders(init_path, destination_path, counter=1):
    init_scale = float(init_path[-4:])
    final_scale = float(destination_path[-4:])

    def resizing(img_path):
        image = cv2.imread(img_path)
        diff = init_scale / final_scale

        new_width = int(image.shape[1] * diff)
        new_height = int(image.shape[0] * diff)

        new_image = cv2.resize(image, (new_width, new_height))
        return new_image

    list_of_image_names = os.listdir(init_path)
    for image_name in list_of_image_names:
        image_path = os.path.join(init_path, image_name)
        new_image = resizing(image_path)
        temp = 'image_' + str(counter) + '.jpg'
        new_image_path = os.path.join(destination_path, temp)
        cv2.imwrite(new_image_path, new_image)
        print(f'image {counter} has been moved')
        counter += 1


def params_count(model):

    total_params = sum(p.numel() for p in model.parameters())
    total_front_params = sum(p.numel() for p in model.frontend.parameters())
    total_back_params = sum(p.numel() for p in model.backend.parameters())
    total_regr_params = sum(p.numel() for p in model.backend.regressor.parameters())
    print(f"total parameters: ", {total_params})
    print(f"total front parameters: ", {total_front_params})
    print(f"total back parameters: ", {total_back_params})
    print(f"total regr parameters: ", {total_regr_params})



    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_front_params = sum(p.numel() for p in model.frontend.parameters() if p.requires_grad)
    train_back_params = sum(p.numel() for p in model.backend.parameters() if p.requires_grad)
    train_regr_params = sum(p.numel() for p in model.backend.regressor.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")
    print(f"train front parameters: ", {train_front_params})
    print(f"train back parameters: ", {train_back_params})
    print(f"train regr parameters: ", {train_regr_params})


def params_count_lite(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {total_trainable_params}')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def custom_loss(output, label):
    return torch.mean(torch.abs(output - label) / label)


class TensorDict(OrderedDict):
    """Container mainly used for dicts of torch tensors. Extends OrderedDict with pytorch functionality."""

    def concat(self, other):
        """Concatenates two dicts without copying internal data."""
        return TensorDict(self, **other)

    def copy(self):
        return TensorDict(super(TensorDict, self).copy())

    def __deepcopy__(self, memodict={}):
        return TensorDict(copy.deepcopy(list(self), memodict))

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorDict\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorDict({n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e for n, e in self.items()})
        return apply_attr

    def attribute(self, attr: str, *args):
        return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

    def apply(self, fn, *args, **kwargs):
        return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorDict, list))
