import json
import os

from PIL import Image
from shared.data_split import DataSplit
import torch
from torch.utils.data import Dataset
from lightning.modules.moco2_module import MocoV2
from lightning.modules.moco2_module_mini import MocoV2Lite

class ClassifierDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                #  image_input: bool = False,
                 max_annotations = -1
                 ):
        
        # if image_input:
        #     raise NotImplementedError("Image input not implemented yet")
        #     self.CSRprocess = MocoV2().load_from_checkpoint(csr_ckpt_path)
        # else:
        print(f"Reading checkpoint at :{csr_ckpt_path}")
        self.CSRprocess = MocoV2Lite().load_from_checkpoint(csr_ckpt_path)
        self.CSRprocess.eval()
        get_csr = lambda resnet_vec: torch.nn.functional.normalize(self.CSRprocess.projection_q(resnet_vec), dim=-1)

        self.root_dir = root_dir
        split_str = {DataSplit.TRAIN:'train', DataSplit.TEST:'test', DataSplit.VAL:'val'}[data_split]
        index_path = os.path.join('/srv/cvmlp-lab/flash1/gchhablani3/housekeep', f'all_scenes_indices.pt')
        print(f'Loading index from {index_path}')
        original_index = torch.load(index_path)
        
        file_dict_reader = lambda fp: json.load(open(os.path.join('/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw', fp.split('|')[0], 
                                                'baseline_phasic_oracle',
                                                'csr',
                                                fp.split('|')[1])))

        get_resnet = lambda fp, obj_1, obj_2 : torch.load((os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.pt')))))
        
        get_image = lambda fp, obj_1, obj_2 : torch.load((os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'images',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.png')))))
        
        self.num_classes = len(original_index['iids'])

        self.data = []
        for o1 in range(5):
            files = original_index['arr'][o1][o1]
            for scene in os.listdir(self.root_dir):
                if scene in ['all_scenes_indices.pt', 'indices_partwise']:
                    continue
                for filename in os.listdir(os.path.join(self.root_dir, 
                                        scene, 
                                        'baseline_phasic_oracle',
                                        'resnet')):
                    # if image_input:
                    #     csr_input = get_image(filename, original_index.index(item['iid']),  original_index.index(item['iid']))
                    # else:
                    if filename.startswith(f'{o1}_{o1}'):
                        csr_input = torch.load(os.path.join(self.root_dir, 
                                            scene, 
                                            'baseline_phasic_oracle',
                                            'resnet', filename))
                        csr_feature = get_csr(csr_input).detach()
                        self.data.append((csr_feature.squeeze(0), torch.tensor([o1]).to(int)))

        self.length = max_annotations if max_annotations > 0 else len(self.data)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, inputs):
        data = {'features': torch.tensor([]), 'labels':torch.tensor([])}
        for feature, obj in inputs:
            data['features'] = torch.cat([data['features'], feature.unsqueeze(0)], dim=0)
            data['labels'] = torch.cat([data['labels'], obj.unsqueeze(0)], dim=0).to(int)
        return data