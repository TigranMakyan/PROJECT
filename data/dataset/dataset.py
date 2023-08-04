import os
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import numpy as np

'''
-data_path
    -pairs
        -forlders with pairs.txt
    -phoenix
        -S6
            -zl548
                -MegaDepth_v1
                    -folders with images
'''

NUM_WORKERS=0

class MegaDepthDataset(Dataset):

    def __init__(self, path_index, data_path='/media/storagedrive/Megadepth/', transform=None) -> None:
        super().__init__()
        print(f'BUILDING MEGADEPTH DATASET FOR {path_index} FOLDER')
        self.transform = transform
        self.images_path = os.path.join(data_path, 'phoenix/S6/zl548/MegaDepth_v1', path_index, 'dense0/imgs')
        self.pairs_path = os.path.join(data_path, 'pairs', path_index, 'sparse-txt/pairs.txt')
        
        self.pairs_file = open(self.pairs_path, 'r')
        self.pairs = self.pairs_file.readlines()

        self.scale_dataset = []
        for item in self.pairs:
            item_info = item.split(' ')
            self.scale_dataset.append([item_info[0], item_info[1], item_info[-1]])

        self.scale_distr = self.build_scale_rep()
        self.scale_levels = 13
        self.scale_base = np.log(np.sqrt(2))


    def __len__(self):
        return len(self.scale_dataset)

    def __getitem__(self, index):
        item = self.scale_dataset[index]
        image1 = cv2.imread(os.path.join(self.images_path, item[0]))
        image2 = cv2.imread(os.path.join(self.images_path, item[1]))
        scale = item[-1]

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, scale

# cfg list size is 183
def create_megadepth_datasets(cfg_train_list, cfg_val_list, cfg_test_list, data_path, transform_train, transform_test):
    '''
    Function for building training, validation and testind datasets
    We used ConcatDataset tool to concate MegaDepth each folders dataset 
    '''
    list_of_train_sets = [MegaDepthDataset(i, transform=transform_train, data_path=data_path) for i in cfg_train_list]
    list_of_val_sets = [MegaDepthDataset(j, transform=transform_test, data_path=data_path) for j in cfg_val_list]
    list_of_test_sets = [MegaDepthDataset(k, transform=transform_test, data_path=data_path) for k in cfg_test_list]

    train_set = ConcatDataset(list_of_train_sets)
    val_set = ConcatDataset(list_of_val_sets)
    test_set = ConcatDataset(list_of_test_sets)

    print(f"Total training images: {len(train_set)}")
    print(f"Total validation images: {len(val_set)}")
    print(f"Total test images: {len(test_set)}")
    print('DATASETS ARE READY')
    return train_set, val_set, test_set



def create_megadepth_dataloaders(train_set, val_set, test_set, batch_size):
    '''
    Building dataloaders for training
    '''
    train_datalaoder = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_datalaoder = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    test_datalaoder = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    print('DATALOADERS ARE READY')
    return train_datalaoder, val_datalaoder, test_datalaoder


    


