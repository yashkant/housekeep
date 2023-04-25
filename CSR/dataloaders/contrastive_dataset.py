import json
import os
import random
import math
import numpy as np
import torch
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from shared.data_split import DataSplit
from shared.utils import get_box
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ContrastiveDataset(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit):
        # set the root directory
        self.root_dir = root_dir
        self.scenes_indices = torch.load(os.path.join(root_dir, 'all_scenes_indices.pt'))
        self.iids = self.scenes_indices['iids']
        self.data_split = data_split

        if data_split == 1:
            # mask = slice(0, len(self.iids)//2)
            mask_by_data_split = slice(0, 8)
        elif data_split == 2:
            # mask = slice(len(self.iids)//2, len(self.iids)//2+len(self.iids)//4)
            mask_by_data_split = slice(8, 12)
        else:
            # mask = slice(len(self.iids)//2+len(self.iids)//4, len(self.iids))
            mask_by_data_split = slice(12, 14)


        #DEBUG: overriding mask for mini dataset
        if 'mini' in root_dir.split('/')[-1]:
            self.mask = slice(0, 5)

        elif 'full' in root_dir.split('/')[-1]:
            self.mask = mask_by_data_split

        else:
            assert False, 'Data path currently {}, should be csr_full or csr_mini'.format(root_dir.split('/')[0])

        self.current_iids = self.iids[self.mask]

        self.scene_indices_array = [[[] for _ in self.iids] for _ in self.iids]

        #TODO: make post post post processing script for this
        for i in range(len(self.iids)): # i is the index to the object iid
            for j in range(len(self.iids)):
                filepath_ij = os.path.join(root_dir, 'indices_partwise', f'{i}_{j}_indices.pt')

                if os.path.exists(filepath_ij):
                    # print(i, j, filepath_ij)

                    data_array_ij = torch.load(filepath_ij)

                    for dp_ij in data_array_ij['arr']:
                        assert isinstance(dp_ij[0], int) or isinstance(dp_ij[0], np.int64), f'{dp_ij[0]} is of type {type(dp_ij[0])}'
                        self.scene_indices_array[i][j].append(dp_ij[0]) # append fidx, ignore bounding boxes

        print(data_split, self.current_iids, self.mask)

        # Note: scene indices array indexed not by index of obj iid NOT obj iid
        self.masked_scene_indices_arr = np.array(self.scene_indices_array)[self.mask, self.mask]
        print('number of files for each pair: ', [len(self.masked_scene_indices_arr[i][j]) for i in range(len(self.current_iids)) for j in range(len(self.current_iids))])
        # print('total: ', sum([len(self.scene_indices_array[i][j]) for i in range(len(self.current_iids)) for j in range(len(self.current_iids))]))

        self.indexed_data = []
        for o1 in range(len(self.current_iids)):
            for o2 in range(len(self.current_iids)):
                files = self.masked_scene_indices_arr[o1][o2]
                for i,f1 in enumerate(files):
                    for f2 in files[i+1:]:
                        f1_name = self.scenes_indices['files'][f1]
                        f2_name = self.scenes_indices['files'][f2]
                        if f1_name.split('|')[0] == f2_name.split('|')[0]: # positive pairs must come from the same scene and episode
                            self.indexed_data.append((o1, o2, f1_name, f2_name)) # all positive pairs
       
        self.length = len(self.indexed_data)

        # save the data augmentations that are to be applied to the images
        self.transform = transform
        assert self.transform is not None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obj_1, obj_2, file_path_1, file_path_2 = self.indexed_data[idx]

        file_dict_resnet = \
            lambda fp: torch.load(os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.pt'))
                                    )
                                )

        data_pair = []
        for file_path in [file_path_1, file_path_2]:

            tensor_data = file_dict_resnet(file_path)
            data_pair.append(tensor_data)

        # create dict and return
        return dict({'input': data_pair[0].detach(), 'is_self_feature': obj_1==obj_2}), dict({'input': data_pair[1].detach(), 'is_self_feature': obj_1==obj_2})

    def collate_fn(self, full_data_list):

        # full_data_list is a list of tuples (q_dict, k_dict) 
        q_tensors = torch.cat([d[0]['input'] for d in full_data_list], dim=0)
        q_self_flags = torch.tensor([d[0]['is_self_feature'] for d in full_data_list], dtype=torch.bool)

        k_tensors = torch.cat([d[1]['input'] for d in full_data_list], dim=0)
        k_self_flags = torch.tensor([d[1]['is_self_feature'] for d in full_data_list], dtype=torch.bool)

        return dict({'input': q_tensors, 'is_self_feature': q_self_flags}), dict({'input': k_tensors, 'is_self_feature': k_self_flags})