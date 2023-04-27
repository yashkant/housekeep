from datetime import datetime
import pickle as pkl

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

torch.manual_seed(2345678)

class MLP(pl.LightningModule):
    def __init__(self, input_size, config):
        super().__init__()

        # 2-layer MLP
        self.fc1 = nn.Linear(input_size, config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.fc3 = nn.Linear(config['hidden_size'], 1)

        self.config = config

        self.loss = nn.BCEWithLogitsLoss()
        self.train_log = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.train_log.append(loss.clone().detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimizer

    def validation_step(self, test_batches, batch_idx):
        self.test_step(test_batches, batch_idx)

    def test_step(self, test_batches, batch_idx):

        self.eval()

        '''
        #DEBUG 
        if len(test_batches) == 2:
            x, y = test_batches
            y_hat = self.forward(x)
            y_pred = torch.sigmoid(y_hat.squeeze(1))
            y_pred = torch.round(y_pred)

            # compute confusion matrix metrics
            tp = torch.sum((y == 1) & (y_pred == 1))
            tn = torch.sum((y == 0) & (y_pred == 0))
            fp = torch.sum((y == 0) & (y_pred == 1))
            fn = torch.sum((y == 1) & (y_pred == 0))
            confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])

            # metrics
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            return f1
        '''

        x, y = test_batches
        y_hat = self.forward(x)
        y_pred = torch.sigmoid(y_hat.squeeze(1))
        y_pred = torch.round(y_pred)

        # return_output = dict({
        #     'y': y.cpu().numpy(),
        #     'y_pred': y_pred.cpu().detach().numpy(),
        #     'user_labels': [data_dict['user_id'] 
        #                         for data_dict in data_dict_list],
        #     'seen_labels': [data_dict['seen_object']
        #                         for data_dict in data_dict_list],
        # })

        # compute confusion matrix metrics
        tp = torch.sum((y == 1) & (y_pred == 1))
        tn = torch.sum((y == 0) & (y_pred == 0))
        fp = torch.sum((y == 0) & (y_pred == 1))
        fn = torch.sum((y == 1) & (y_pred == 0))
        confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        self.log('val_prec', precision, on_step=True, on_epoch=True, logger=True)
        self.log('val_recl', recall, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, logger=True)

        # self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        return confusion_matrix, dict({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        })

