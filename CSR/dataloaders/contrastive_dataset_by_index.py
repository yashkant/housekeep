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

    def __init__(self, root_dir, transform, data_split: DataSplit, test_unseen_objects = True):
        # set the root directory
        self.root_dir = root_dir
        self.scenes_indices = torch.load(os.path.join(root_dir, 'all_scenes_indices.pt'))
        self.iids = self.scenes_indices['iids']
        self.data_split = data_split

        if data_split == DataSplit.TRAIN:
            mask_by_data_split = slice(0, 40)
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.VAL:
            mask_by_data_split = slice(38, 42)
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.TEST:
            if test_unseen_objects:
                mask_by_data_split = slice(59, 100)
            else:
                mask_by_data_split = slice(0, 100)
            use_episode = lambda e: e <= 10
        else:
            assert False, 'Data split not recognized'
        
        self.mask = mask_by_data_split
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
                        scene1 = f1_name.split('|')[0]
                        scene2 = f2_name.split('|')[0]
                        episode1 = int(f1_name.split('|')[1].split('.')[0].split('_')[-1])//1000
                        episode2 = int(f2_name.split('|')[1].split('.')[0].split('_')[-1])//1000
                        # print(scene1, scene2, int(f1_name.split('|')[1].split('.')[0].split('_')[-1]), int(f2_name.split('|')[1].split('.')[0].split('_')[-1]), episode1, episode2)
                        # raise Exception('stop')
                        if scene1 == scene2 and episode1 == episode2 and use_episode(episode1): # positive pairs must come from the same scene and episode
                            o1_original = self.iids.index(self.current_iids[o1])
                            o2_original = self.iids.index(self.current_iids[o2])
                            if os.path.exists(self.file_path_resnet(f1_name, o1_original, o2_original)) and os.path.exists(self.file_path_resnet(f2_name, o1_original, o2_original)):
                                self.indexed_data.append((o1_original, o2_original, f1_name, f2_name)) # all positive pairs
                            else:
                                print('missing file: ', self.file_path_resnet(f1_name, o1_original, o2_original), self.file_path_resnet(f2_name, o1_original, o2_original))
                                raise Exception('missing file')
       
        self.length = len(self.indexed_data)
        print('Total length of dataset type ', data_split, ': ', self.length)

        # # save the data augmentations that are to be applied to the images
        # self.transform = transform
        # assert self.transform is not None

    def file_path_resnet(self, fp, o1, o2): 
            filepath_full = os.path.join(self.root_dir, 
                                        fp.split('|')[0], 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_{}'.format(o1, o2, fp.split('|')[1].replace('.json','.pt'))
                                    )
            return filepath_full

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obj_1, obj_2, file_path_1, file_path_2 = self.indexed_data[idx]

        file_dict_resnet = lambda fp: torch.load(self.file_path_resnet(fp, obj_1, obj_2))

        data_pair = []
        for file_path in [file_path_1, file_path_2]:
            tensor_data = file_dict_resnet(file_path)
            if tensor_data is not None:
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