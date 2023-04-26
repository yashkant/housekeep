import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, ConfusionMatrix


class FeatureDecoderModule(pl.LightningModule):
    def __init__(self, 
                feature_size: int = 512,
                num_classes: int = 2,
                image_input: bool = False):
        super().__init__()

        if image_input:
            raise NotImplementedError("Image input not implemented yet")
        self.linear = torch.nn.Linear(feature_size, num_classes)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_confmat = ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = ConfusionMatrix(num_classes=num_classes)
        self.val_misclass = {}


    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature)
        loss = torch.nn.CrossEntropyLoss()(pred, target.squeeze(-1))

        acc = self.train_acc(torch.argmax(pred, dim=1), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature)
        loss = torch.nn.CrossEntropyLoss()(pred, target.squeeze(-1))

        acc = self.val_acc(torch.argmax(pred, dim=1), target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss


    def test_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature)
        loss = torch.nn.CrossEntropyLoss()(pred, target.squeeze(-1))

        acc = self.val_acc(torch.argmax(pred, dim=1), target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
