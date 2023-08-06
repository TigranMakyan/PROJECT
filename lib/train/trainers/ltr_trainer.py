import os
# import sys
# sys.path.insert(0, '.')
from collections import OrderedDict
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
from lib.train.trainers.base_trainer import BaseTrainer
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler

class LTRTRainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False) -> None:
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        self.stats = OrderedDict({loader.name: None for loader in loaders})

        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}
        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    
    def cycle_dataset(self, loader):
        '''Do a cycle of training or validation.'''
        pass

    def train_epoch(self):
        '''Do one epoch for each loader'''
        pass

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        '''Inintialize stats if not initialized yet'''
        pass

    def _print_stats(self, i, loader, batch_size):
        pass

    def _stats_new_epoch(self):
        '''Record learning rate'''
        pass

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)


