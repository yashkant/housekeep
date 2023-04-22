import json
import os
import random
from shared.utils import get_box
from shared.constants import CLASSES_TO_IGNORE

from PIL import Image
from shared.data_split import DataSplit
import torch
from torch.utils.data import Dataset
from lightning.modules.moco2_module import Moco2Module
from lightning.modules.moco2_module_mini import MocoV2Lite


class ReceptacleDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                 image_input: bool = False,
                 max_annotations = -1
                 ):
        
        if image_input:
            raise NotImplementedError("Image input not implemented yet")
            self.CSRprocess = Moco2Module().load_from_checkpoint(csr_ckpt_path)
        else:
            self.CSRprocess = MocoV2Lite().load_from_checkpoint(csr_ckpt_path)
        
        split_str = {DataSplit.TRAIN:'train', DataSplit.TEST:'test', DataSplit.VAL:'val'}[data_split]
        index_path = os.path.join(root_dir, '..', f'{split_str}_indices.pt')
        print(f'Loading index from {index_path}')
        original_index = torch.load(index_path)
        
        file_dict = lambda fp: json.load(open(os.path.join(root_dir, fp.split('|')[0], 
                                                'baseline_phasic_oracle',
                                                'csr',
                                                fp.split('|')[1])))

        get_resnet = lambda fp, obj_1, obj_2 : torch.load(open(os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.pt')))))
        
        get_image = lambda fp, obj_1, obj_2 : torch.load(open(os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'images',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.png')))))
        
        self.data = []
        for file in original_index['files']:
            items = file_dict['items']
            gt_mapping = file_dict['current_mapping']
            for item1 in [i for i in items if i['type'] == 'obj']:
                for item2 in [i for i in items if i['type'] == 'rec']:
                    if item1 != item2:
                        if image_input:
                            csr_input = get_image(file, original_index.index(item1['iid']),  original_index.index(item2['iid']))
                        else:
                            csr_input = get_resnet(file, original_index.index(item1['iid']),  original_index.index(item2['iid']))
                        csr_feature = self.CSRprocess(csr_input)
                        if gt_mapping[item1['obj_key']] == item2['obj_key']:
                            label = 1
                        else:
                            label = 0
                        self.data.append((csr_feature, label))

        self.length = max_annotations if max_annotations > 0 else len(self.data)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]
