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
            mask = slice(0, 8)
        elif data_split == 2:
            # mask = slice(len(self.iids)//2, len(self.iids)//2+len(self.iids)//4)
            mask = slice(8, 12)
        else:
            # mask = slice(len(self.iids)//2+len(self.iids)//4, len(self.iids))
            mask = slice(12, 14)
        
        self.mask = mask
        self.current_iids = self.iids[mask]
        # self.current_iids_indices = [self.iids.index(curr_id) for curr_id in self.current_iids]
        # print(self.current_iids_indices)
        # print(mask)
        # print(len(self.current_iids))
        self.masked_scene_indices_arr = self.scenes_indices['arr'][mask, mask, :]

        self.indexed_data = []
        for o1 in range(len(self.current_iids)):
            for o2 in range(len(self.current_iids)):
                files = np.argwhere(self.masked_scene_indices_arr[o1,o2,:][0] == 1).reshape(-1)
                for i,f1 in enumerate(files):
                    for f2 in files[i+1:]:
                        if f1.split('|')[0] == f2.split('|')[0]: # positive pairs must come from the same scene and episode
                            self.indexed_data.append((o1, o2, f1, f2)) # all positive pairs
       
        self.length = len(self.indexed_data)

        # save the data augmentations that are to be applied to the images
        self.transform = transform
        assert self.transform is not None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obj_1, obj_2, file_path_1, file_path_2 = self.indexed_data[idx]

        file_dict_resnet = \
            lambda fp: torch.load(open(os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_{}'.format(obj_1, obj_2, fp.split('|')[1].replace('.json','.pt')))))

        data_pair = []
        for file_path_tuple in [file_path_1, file_path_2]:

            print(f'file tuple: {file_path_tuple}')

            file_path_idx = file_path_tuple[0] # file_path_tuple = (fidx, bb1, bb2)
            tensor_data = file_dict_resnet(self.scenes_indices['files'][file_path_idx])

            print(f'shape of resnet tensor: {tensor_data.size()}')
            input('wait')
        
            data_pair.append(tensor_data)

        # create dict and return
        return data_pair[0], data_pair[1]
