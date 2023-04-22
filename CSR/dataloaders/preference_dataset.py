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
from transformers import CLIPImageProcessor

def get_preference(item1, item2, episode_map, persona=None):
    if persona is not None:
        raise NotImplementedError("Persona not implemented yet")
    else:
        if item2['obj_key'] in episode_map[item1['obj_key']]:
            label = 1
        else:
            label = 0
    return label

class PreferenceDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                 image_input: bool = False,
                 max_annotations = -1,
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
        
        ## TODO: Varify that this is the correct way to get the CLIP features
        get_clip_feature = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def get_features(file, item):
            if image_input:
                csr_input = get_image(file, original_index.index(item['iid']),  original_index.index(item['iid']))
            else:
                csr_input = get_resnet(file, original_index.index(item['iid']),  original_index.index(item['iid']))
            csr_feature = self.CSRprocess(csr_input)
            clip_feature = get_clip_feature(item['cropped_image']).data['pixel_values'][0].reshape(-1)
            return csr_feature, torch.from_numpy(clip_feature)

        self.data = []
        for file in original_index['files']:
            items = file_dict['items']
            gt_mapping = file_dict['correct_mapping']
            for item1 in [i for i in items if i['type'] == 'obj']:
                csr_feature1, clip_feature1 = get_features(file, item1)
                for item2 in [i for i in items if i['type'] == 'rec']:
                    if item1 != item2:
                        csr_feature2, clip_feature2 = get_features(file, item2)
                        label = get_preference(item1, item2, gt_mapping, persona=None)
                        self.data.append(((csr_feature1, clip_feature1, csr_feature2, clip_feature2), label))
                        
        self.length = max_annotations if max_annotations > 0 else len(self.data)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns : Tuple (data,label)
                    data : (csr_feature1, clip_feature1, csr_feature2, clip_feature2)
                    label: 0 or 1
        """
        return self.data[idx]
