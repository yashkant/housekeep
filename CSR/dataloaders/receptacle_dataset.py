import json
import os

from PIL import Image
from shared.data_split import DataSplit
import torch
from torch.utils.data import Dataset
from lightning.modules.moco2_module import MocoV2
from lightning.modules.moco2_module_mini import MocoV2Lite


class ReceptacleDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                #  image_input: bool = False,
                 max_annotations = -1,
                 test_unseen_objects = True
                 ):
        
        # if image_input:
        #     raise NotImplementedError("Image input not implemented yet")
        #     self.CSRprocess = MocoV2().load_from_checkpoint(csr_ckpt_path)
        # else:

        if data_split == DataSplit.TRAIN:
            use_iid_idx = lambda o: o<40
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.VAL:
            use_iid_idx = lambda o: o>=38 and o<42
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.TEST:
            if test_unseen_objects:
                use_iid_idx = lambda o: o>=59
            else:
                use_iid_idx = lambda o: True
            use_episode = lambda e: e <= 10
        else:
            assert False, 'Data split not recognized'

        print(f"Reading checkpoint at :{csr_ckpt_path}")
        self.CSRprocess = MocoV2Lite().load_from_checkpoint(csr_ckpt_path)
        self.CSRprocess.eval()
        get_csr = lambda resnet_vec: torch.nn.functional.normalize(self.CSRprocess.projection_q(resnet_vec), dim=-1)

        split_str = {DataSplit.TRAIN:'train', DataSplit.TEST:'test', DataSplit.VAL:'val'}[data_split]
        index_path = os.path.join(root_dir, f'all_scenes_indices.pt')
        print(f'Loading index from {index_path}')
        original_index = torch.load(index_path)
        
        file_dict_reader = lambda fp: json.load(open(os.path.join('/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw', fp.split('|')[0], 
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
            episode = int(file.split('|')[1].split('.')[0].split('_')[-1])//1000
            if not use_episode(episode):
                continue
            file_dict = file_dict_reader(file)
            items = file_dict['items']
            gt_mapping = file_dict['current_mapping']
            for item1 in [i for i in items if i['type'] == 'obj']:
                if use_iid_idx(item1['iid']):
                    for item2 in [i for i in items if i['type'] == 'rec']:
                        if use_iid_idx(item1['iid']):
                            # if image_input:
                            #     csr_input = get_image(file, original_index.index(item1['iid']),  original_index.index(item2['iid']))
                            # else:
                            csr_input = get_resnet(file, original_index.index(item1['iid']),  original_index.index(item2['iid']))
                            csr_feature = get_csr(csr_input)
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
