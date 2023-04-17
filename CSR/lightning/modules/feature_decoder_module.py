import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, ConfusionMatrix


class FeatureDecoderModule(pl.LightningModule):
    def __init__(self, 
                feature_size: int = 512,
                num_classes: int = 2):
        super().__init__()

        self.linear = torch.nn.Linear(feature_size, num_classes)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.test_confmat = ConfusionMatrix(num_classes=len(self.conf.classes))
        self.val_misclass = {}


    def forward(self, x):
        return self.linear(self.encoder(x))


    def training_step(self, batch, batch_idx):
        feature, target = batch
        pred = self(feature)
        loss = F.cross_entropy(pred, target)

        acc = self.train_acc(torch.argmax(pred, dim=1), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x_dict, target = batch
        x = torch.cat((x_dict['image'], x_dict['mask_1'], x_dict['mask_2']), 1)
        pred = self(x)
        loss = F.cross_entropy(pred, target)

        flat_preds = torch.argmax(pred, dim=1)
        acc = self.val_acc(flat_preds, target)
        self.val_confmat(flat_preds, target)
        misclass_indicator = flat_preds != target
        indices = torch.arange(x.shape[0])

        self.val_misclass[batch_idx] = [indices[misclass_indicator],
                                        flat_preds[misclass_indicator], target[misclass_indicator]]
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        x_dict, target = batch
        x = torch.cat((x_dict['image'], x_dict['mask_1'], x_dict['mask_2']), 1)
        pred = self(x)
        loss = F.cross_entropy(pred, target)

        flat_preds = torch.argmax(pred, dim=1)
        acc = self.test_acc(flat_preds, target)
        self.test_confmat(flat_preds, target)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
