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
        self.fc3 = nn.Linear(config['hidden_size'], config['output_size'])

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
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_log.append(loss.clone().detach().cpu().numpy())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        return optimizer

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

        x, y, data_dict_list = test_batches
        y_hat = self.forward(x)
        y_pred = torch.sigmoid(y_hat.squeeze(1))
        y_pred = torch.round(y_pred)

        return_output = dict({
            'y': y.cpu().numpy(),
            'y_pred': y_pred.cpu().detach().numpy(),
            'user_labels': [data_dict['user_id'] 
                                for data_dict in data_dict_list],
            'seen_labels': [data_dict['seen_object']
                                for data_dict in data_dict_list],
        })

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

        # self.log('test_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        return confusion_matrix, dict({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        }), return_output


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, user_conditioned=False, is_train=True, device=torch.device('cuda')):

        self.data_dict = data
        self.device = device

        # self.test_data = self.data_dict['test']
        # self.test_data_indextokey = dict({
        #     i: k for i, k in enumerate(self.test_data.keys())
        #     })

        self.is_train = is_train
        self.user_conditioned = user_conditioned

        self.input_tensor_dim = \
            self.data_dict[0]['csr_item_1'].shape[-1] + \
            self.data_dict[0]['clip_item_1'].shape[-1] + \
            self.data_dict[0]['csr_item_2'].shape[-1] + \
            self.data_dict[0]['clip_item_2'].shape[-1] + \
            self.data_dict[0]['room_embb'].shape[-1]

        if self.user_conditioned:
            self.input_tensor_dim += \
                self.data_dict[0]['persona_embb'].shape[-1]

    def __getitem__(self, index):

        return self.data_dict[index]
    
    def __len__(self):

        return len(self.data_dict)

    def return_input_dim(self):
        return self.input_tensor_dim

    def collate_fn(self, data_points):
        ''' Collate function for dataloader.
            Args:
                data_points: A list of dicts'''

        # data_points is a list of dicts
        csr_embbs_1 = torch.cat([torch.tensor(d['csr_item_1']).unsqueeze(0) for d in data_points], dim=0)
        clip_embbs_1 = torch.cat([torch.tensor(d['clip_item_1']) for d in data_points], dim=0)
        csr_embbs_2 = torch.cat([torch.tensor(d['csr_item_2']).unsqueeze(0) for d in data_points], dim=0)
        clip_embbs_2 = torch.cat([torch.tensor(d['clip_item_2']) for d in data_points], dim=0)

        room_embbs = torch.cat([torch.tensor(d['room_embb']).unsqueeze(0) for d in data_points], dim=0)

        if self.user_conditioned:

            user_embbs = torch.cat([torch.tensor(d['persona_embb']).unsqueeze(0) for d in data_points], dim=0)

            input_tensor = torch.cat([csr_embbs_1, clip_embbs_1, csr_embbs_2, clip_embbs_2, room_embbs, user_embbs], dim=1)

        else:

            input_tensor = torch.cat([csr_embbs_1, clip_embbs_1, csr_embbs_2, clip_embbs_2, room_embbs], dim=1)

        labels = torch.tensor([d['label'] for d in data_points])

        if self.is_train:
            return input_tensor.to(self.device), \
                labels.type(torch.float).to(self.device)

        else:
            return input_tensor.to(self.device), \
                    labels.type(torch.float).to(self.device), \
                    data_points


def main():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m-%Y_%H-%M-%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'hidden_size': 512,
        'output_size': 1,
        'batch_size': 32,
        'max_epochs': 15,
        'lr': 1e-4,
        'num_layers': 2,
        'weight_decay': 1e-6,
        'train_data_path': '/coc/flash5/mpatel377/data/csr_clip_preferences/seen_partial_preferences_4600.pt',
        'test_data_path': None,
        'user_conditioned': True
    }

    print('config: ')
    for k, v in config.items():
        print(f'{k}: {v}')

    # load data
    train_data_dict = torch.load(config['train_data_path'])

    # create dataset class
    train_dataset = Dataset(train_data_dict, user_conditioned=config['user_conditioned'], is_train=True, device=device)
    # test_dataset = Dataset(data_dict, user_conditioned=config['user_conditioned'], is_train=False, device=device)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                collate_fn=train_dataset.collate_fn)

    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, 
    #                             collate_fn=test_dataset.collate_fn)

    print('data loaders are ready')

    # instantiate model
    model = MLP(input_size=train_dataset.return_input_dim(), config=config)
    print('model is ready')

    # # preload test batch
    # test_batch = next(iter(test_loader))

    # # before training
    # model.to(device)
    # _, test_metrics, _ = model.test_step(test_batch, 0)

    # print('Test f1 (before training): ', test_metrics['f1'].cpu().numpy())

    # training loop
    trainer = pl.Trainer(max_epochs=config['max_epochs'])
    trainer.fit(model, train_loader)

    # after training

    '''
    # Testing on full train set
    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, 
                                collate_fn=train_dataset.collate_fn)
    model.to(device)
    full_train_batch = next(iter(full_train_loader))
    train_f1 = model.test_step(full_train_batch, 0)
    print('f1 train: ', train_f1.cpu().numpy())
    '''

    # # testing loop 
    # model.to(device)
    # confusion_matrix, test_metrics, output = model.test_step(test_batch, 0)

    # # final stats
    # print('Test f1 (after training): ', test_metrics['f1'].cpu().numpy())
    # print('Test data confusion matrix: ', confusion_matrix.cpu().numpy())

    # # save results
    # with open('results_{}.pkl'.format(timestampStr), 'wb') as f:
    #     pkl.dump(dict({
    #         'config:': config,
    #         'train_loss_history': model.train_log,
    #         'test_confusion_matrix': confusion_matrix.cpu().numpy(),
    #         'test_metrics': test_metrics,
    #         'network_output': output
    #     }), f)

    for l in model.train_log:
        print(l)

if __name__ == '__main__':
    main()
