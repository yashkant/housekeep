import itertools
import json
import os
import random
import math

import numpy as np
import torch
from PIL import Image
from shared.constants import CLASSES_TO_IGNORE, DATALOADER_BOX_FRAC_THRESHOLD, IMAGE_SIZE
from shared.data_split import DataSplit
from shared.utils import get_box
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class ContrastiveDataset(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit):
        # set the root directory
        self.root_dir = root_dir
        self.scenes_indices = torch.load(os.path.join(root_dir, 'all_scenes_indices.pt'))
        self.iids = self.scenes_indices['iids']
        self.data_split = data_split

        if data_split == 1:
            mask = slice(0, len(self.iids)//2)
        elif data_split == 2:
            mask = slice(len(self.iids)//2, len(self.iids)//2+len(self.iids)//4)
        else:
            mask = slice(len(self.iids)//2+len(self.iids)//4, len(self.iids))
        
        self.mask = mask
        self.current_iids = self.iids[mask]
        # self.current_iids_indices = [self.iids.index(curr_id) for curr_id in self.current_iids]
        # print(self.current_iids_indices)
        # print(mask)
        # print(len(self.current_iids))
        self.masked_scene_indices_arr = self.scenes_indices['arr'][mask, mask, :]
        # print(self.masked_scene_indices_arr.shape)
        # print(self.masked_scene_indices_arr.shape)
        num_of_scenes_per_iid_pair = np.sum(self.masked_scene_indices_arr, axis=-1)

        possible_pair_combinations = (num_of_scenes_per_iid_pair * (num_of_scenes_per_iid_pair - 1))/ 2

        self.total_pairs = np.cumsum(possible_pair_combinations.flatten()).reshape(possible_pair_combinations.shape)
        self.pos_to_neg_ratio = 0.5

        # save the data augmentations that are to be applied to the images
        self.transform = transform
        assert self.transform is not None

    def __len__(self):
        return math.floor(self.total_pairs[-1, -1]/self.pos_to_neg_ratio)

    def __getitem__(self, idx):
        
        if random.random() < self.pos_to_neg_ratio:
            pos = True
            # Get the pair at that index
            if (idx < self.total_pairs[0, 0]):
                obj_1, obj_2 = 0, 0
                file_pair_index = idx
            else:
                obj_1, obj_2 = np.argwhere(self.total_pairs < idx)[-1]
                # Get the pair at the previous (this helps us get the start of range of indices which has obj_1 and obj_2)
                prev_obj_1, prev_obj_2 = np.argwhere(self.total_pairs < idx)[-2]
                # Gives us index in the files list for current pair
                file_pair_index = idx - self.total_pairs[prev_obj_1, prev_obj_2]
     
            # Where in the array the files are there
            files_list = np.argwhere(self.masked_scene_indices_arr[obj_1, obj_2, :] == 1).squeeze()
            # Pairs of files containing the same object pairs
            files_pair_list = [[(i,j) for j in range(i+1,len(files_list))] for i in range(len(files_list))] 
            # Get the file pair 
            file_path_1 = self.masked_scene_indices_arr[obj_1, obj_2, files_pair_list[file_pair_index][0]]
            file_path_2 = self.masked_scene_indices_arr[obj_1, obj_2, files_pair_list[file_pair_index][1]]

        else:
            # negative sampling
            pos = False
            # First Sample
            # Get the pair at that index
            if (idx < self.total_pairs[0, 0]):
                obj_1, obj_2 = 0, 0
                file_pair_index = idx
            else:
                obj_1, obj_2 = np.argwhere(self.total_pairs < idx)[-1]
                # Get the pair at the previous (this helps us get the start of range of indices which has obj_1 and obj_2)
                prev_obj_1, prev_obj_2 = np.argwhere(self.total_pairs < idx)[-2]
                # Gives us index in the files list for current pair
                file_pair_index = idx - self.total_pairs[prev_obj_1, prev_obj_2]
            
            # Where in the array the files are there
            files_list = np.argwhere(self.masked_scene_indices_arr[obj_1, obj_2, :] == 1)
            # Pairs of files containing the same object pairs
            # Get the file pair 
            file_path_1 = self.masked_scene_indices_arr[obj_1, obj_2, random.choice(files_list)]

            # Second Sample
            # Get the total number of objects
            num_objs = len(self.current_iids)
            # Initialize neg_obj_1 and neg_obj_2 to None
            neg_obj_1 = neg_obj_2 = None
            # TODO: Better way would be to sample negatives directly from pairs!
            # Loop until we get two objects that are not the same as obj_1 and obj_2, and not already present in neg_obj_1 and neg_obj_2
            while neg_obj_1 is None or (neg_obj_1, neg_obj_2) == (obj_1, obj_2) or self.masked_scene_indices_arr[neg_obj_1, neg_obj_2].sum()==0:
                # Randomly select two objects
                neg_obj_1, neg_obj_2 = np.random.randint(num_objs, size=(2,))
            
            # Where in the array the files are there for neg_obj_1 and neg_obj_2
            neg_files_list = np.argwhere(self.masked_scene_indices_arr[neg_obj_1, neg_obj_2, :] == 1)
            # Get the file pair 
            file_path_2 = self.masked_scene_indices_arr[neg_obj_1, neg_obj_2, random.choice(neg_files_list)]

        data_pair = []

        if pos:
            path_2_obj_1, path_2_obj_2 = obj_1, obj_2
        else:
            path_2_obj_1, path_2_obj_2 = neg_obj_1, neg_obj_2

        for file_path_idx, object_tuple in zip([file_path_1, file_path_2],[(obj_1, obj_2), (path_2_obj_1, path_2_obj_2)]):
            file_path = self.scenes_indices['files'][int(file_path_idx[0])]
            with open(os.path.join('/srv/flash1/gchhablani3/housekeep/csr_raw/beechwood_0_int/baseline_phasic_oracle/csr', file_path.split('|')[-1])) as f:
                data_pair_file = json.load(f)

            item_obj_1 = [item for item in data_pair_file['items'] if item['iid']==self.current_iids[object_tuple[0]]]
            item_obj_2 = [item for item in data_pair_file['items'] if item['iid']==self.current_iids[object_tuple[1]]]

            # print(self.data_split, pos)
            # print([item['iid'] for item in data_pair_file['items']]) # All IIDs
            # print(self.current_iids[object_tuple[0]], self.current_iids[object_tuple[1]])
            m1 = get_box(item_obj_1[0]['bounding_box'].reshape(2, 2))
            m2 = get_box(item_obj_2[0]['bounding_box'].reshape(2, 2))

            data = {'mask_1': m1, 'mask_2': m2, 'image': data_pair_file['rgb'], 'is_self_feature': True}
            self.transform(data)
        
            data_pair.append(data)

        # create dict and return
        return data_pair[0], data_pair[1]
