import os
import pytorch_lightning as pl
import dataloaders.augmentations as A
from dataloaders.receptacle_dataset import ReceptacleDataset
from shared.constants import (COLOR_JITTER_BRIGHTNESS,
                                  COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                                  COLOR_JITTER_SATURATION, DEFAULT_NUM_WORKERS,
                                  GRAYSCALE_PROBABILITY, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD, ROTATIONS)
from shared.data_split import DataSplit
from torch.utils.data import DataLoader

class ReceptacleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir, csr_ckpt_dir, drop_last=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.drop_last = drop_last
        csr_ckpt_dir = os.path.join(csr_ckpt_dir, 'model')
        best_ckpt = [f for f in os.listdir(csr_ckpt_dir) if f.endswith('.ckpt')]
        best_ckpt.sort(key=lambda x: float(x.split('.')[0].split('-')[-1].split('=')[1]))
        best_ckpt = best_ckpt[0]
        self.csr_ckpt_path = os.path.join(csr_ckpt_dir, best_ckpt)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = ReceptacleDataset(
                root_dir = self.data_dir, data_split = DataSplit.TRAIN, csr_ckpt_path = self.csr_ckpt_path)
            self.val_set = ReceptacleDataset(
                root_dir = self.data_dir, data_split = DataSplit.VAL, csr_ckpt_path = self.csr_ckpt_path)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = ReceptacleDataset(
                root_dir = self.data_dir, data_split = DataSplit.TEST, csr_ckpt_path = self.csr_ckpt_path)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=DEFAULT_NUM_WORKERS, pin_memory=True, drop_last=self.drop_last)
