import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from lightning.data_modules.ranking_data_module import \
    RankingDataModule
from lightning.modules.ranking_module import MLP
from lightning.custom_callbacks import ConfusionLogger


class RankingTrainer(object):
    def __init__(self, conf):

        self.conf = conf
        seed_everything(self.conf.seed)

    def run(self):
        # Init our data pipeline
        dm = RankingDataModule(self.conf.batch_size, self.conf.data_path, self.conf.checkpoint_path)

        # To access the x_dataloader we need to call prepare_data and setup.
        dm.prepare_data()
        dm.setup()

        # Init our model
        model = MLP(dm.feature_size, self.conf)

        wandb_logger = WandbLogger(project=self.conf.project_name,
                                   name=self.conf.experiment_name+'_ranking_clip',
                                   job_type='train')
        from datetime import datetime
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%m-%d_%H-%M")

        # defining callbacks
        checkpoint_callback = ModelCheckpoint(dirpath=self.conf.checkpoint_path,
                                              filename='rank_pred_clip_'+timestampStr+'/model-{epoch}-{val_acc:.2f}',
                                              verbose=True,
                                              mode='min',
                                              every_n_val_epochs=1)
        learning_rate_callback = LearningRateMonitor(logging_interval='epoch')

        # confusion_callback = ConfusionLogger(self.conf.classes)

        # set up the trainer
        trainer = pl.Trainer(max_epochs=self.conf.epochs,
                             check_val_every_n_epoch=1,
                             progress_bar_refresh_rate=self.conf.progress_bar_refresh_rate,
                             gpus=self.conf.gpus,
                             logger=wandb_logger,
                             callbacks=[learning_rate_callback,
                                        checkpoint_callback,
                                        # confusion_callback
                                        ],
                             checkpoint_callback=True,
                             num_sanity_val_steps=2)

        # Train the model
        trainer.fit(model, dm)

        # Evaluate the model on the held out test set
        trainer.test()

        # Close wandb run
        wandb.finish()
