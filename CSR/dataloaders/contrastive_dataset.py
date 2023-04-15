import itertools
import json
import os
import random
import math
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
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
            # mask = slice(0, len(self.iids)//2)
            mask = slice(0, 5)
        elif data_split == 2:
            # mask = slice(len(self.iids)//2, len(self.iids)//2+len(self.iids)//4)
            mask = slice(5, 6)
        else:
            # mask = slice(len(self.iids)//2+len(self.iids)//4, len(self.iids))
            mask = slice(5, 7)
        
        mask = slice(0, len(self.iids))
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

        # if random.random() < self.pos_to_neg_ratio:
        if idx < math.floor(self.total_pairs[-1, -1]):
            assert idx < math.floor(self.total_pairs[-1, -1])
            pos = True
            # Get the pair at that index
            if (idx < self.total_pairs[0, 0]):
                obj_1, obj_2 = 0, 0
                prev_obj_1, prev_obj_2 = None, None
                file_pair_index = idx
            else:
                try:
                    obj_1, obj_2 = np.argwhere(self.total_pairs > idx)[0]
                except:
                    print(idx, self.total_pairs[0, 0], self.total_pairs[-1, -1])
                    raise()
                # Get the pair at the previous (this helps us get the start of range of indices which has obj_1 and obj_2)
                prev_obj_1, prev_obj_2 = np.argwhere(self.total_pairs <= idx)[-1]
                # Gives us index in the files list for current pair
                file_pair_index = int(idx - self.total_pairs[prev_obj_1, prev_obj_2])
     
            # Where in the array the files are there
            files_list = np.argwhere(self.masked_scene_indices_arr[obj_1, obj_2, :] == 1).reshape(-1)
            # Pairs of files containing the same object pairs
            files_pair_list = [[(i,j) for j in range(i+1,len(files_list))] for i in range(len(files_list))]
            files_pair_list_flatten = []
            for item in files_pair_list:
                files_pair_list_flatten += item
            # Get the file pair

            if prev_obj_1 is not None:
                assert len(files_pair_list_flatten) == self.total_pairs[obj_1, obj_2] - self.total_pairs[prev_obj_1, prev_obj_2]
            else:
                assert len(files_pair_list_flatten) == self.total_pairs[obj_1, obj_2]
            assert file_pair_index < len(files_pair_list_flatten), f"Obj 1: {obj_1}, Obj 2: {obj_2}, Prev Obj 1: {prev_obj_1}, Prev Obj 2: {prev_obj_2}, File List Len: {len(files_list)}, File Pair Index: {file_pair_index}, File Pair List Flatten: {len(files_pair_list_flatten)}"
            file_path_1 = files_list[files_pair_list_flatten[file_pair_index][0]]
            file_path_2 = files_list[files_pair_list_flatten[file_pair_index][1]]

            assert self.masked_scene_indices_arr[obj_1, obj_2, file_path_1] == 1, "File Path 1 does not include objects"
            assert self.masked_scene_indices_arr[obj_1, obj_2, file_path_2] == 1, "File Path 2 does not include objects"

        else:
            idx = idx % int(self.total_pairs[-1, 1])
            # negative sampling
            pos = False
            # First Sample
            # Get the pair at that index
            if (idx < self.total_pairs[0, 0]):
                obj_1, obj_2 = 0, 0
                file_pair_index = idx
            else:
                try:
                    obj_1, obj_2 = np.argwhere(self.total_pairs > idx)[0]
                except:
                    print(idx, self.total_pairs[0, 0], self.total_pairs[-1, -1])
                    raise()
                # Get the pair at the previous (this helps us get the start of range of indices which has obj_1 and obj_2)
                prev_obj_1, prev_obj_2 = np.argwhere(self.total_pairs <= idx)[-1]
                # Gives us index in the files list for current pair
                file_pair_index = idx - self.total_pairs[prev_obj_1, prev_obj_2]
            
            # Where in the array the files are there
            files_list = np.argwhere(self.masked_scene_indices_arr[obj_1, obj_2, :] == 1).reshape(-1)
            # Pairs of files containing the same object pairs
            # Get the file pair 
            file_path_1 = random.choice(files_list)

            # Second Sample
            # Get the total number of objects
            num_objs = len(self.current_iids)
            # Initialize neg_obj_1 and neg_obj_2 to None
            neg_obj_1 = neg_obj_2 = None
            # TODO: Better way would be to sample negatives directly from pairs!
            # Loop until we get two objects that are not the same as obj_1 and obj_2, and not already present in neg_obj_1 and neg_obj_2
            while neg_obj_1 is None or (neg_obj_1, neg_obj_2) == (obj_1, obj_2) or self.masked_scene_indices_arr[neg_obj_1, neg_obj_2, :].sum()==0:
                # Randomly select two objects
                neg_obj_1, neg_obj_2 = np.random.randint(num_objs, size=(2,))
            
            # Where in the array the files are there for neg_obj_1 and neg_obj_2
            neg_files_list = np.argwhere(self.masked_scene_indices_arr[neg_obj_1, neg_obj_2, :] == 1).reshape(-1)
            # Get the file pair
            # print(neg_files_list, type(neg_files_list))
            # print(len(neg_files_list))
            file_path_2 = random.choice(neg_files_list)

            assert self.masked_scene_indices_arr[obj_1, obj_2, file_path_1] == 1, "File Path 1 does not include objects"
            assert self.masked_scene_indices_arr[neg_obj_1, neg_obj_2, file_path_2] == 1, "File Path 2 does not include objects"


        data_pair = []

        if pos:
            path_2_obj_1, path_2_obj_2 = obj_1, obj_2
        else:
            path_2_obj_1, path_2_obj_2 = neg_obj_1, neg_obj_2

        for file_path_idx, object_tuple in zip([file_path_1, file_path_2],[(obj_1, obj_2), (path_2_obj_1, path_2_obj_2)]):
            # print(self.data_split, pos, file_path_idx)
            file_path = self.scenes_indices['files'][file_path_idx]
            with open(os.path.join('/srv/flash1/gchhablani3/housekeep/csr_raw/beechwood_0_int/baseline_phasic_oracle/csr', file_path.split('|')[-1])) as f:
                data_pair_file = json.load(f)
            item_obj_1 = [item for item in data_pair_file['items'] if item['iid']==self.current_iids[object_tuple[0]]]
            item_obj_2 = [item for item in data_pair_file['items'] if item['iid']==self.current_iids[object_tuple[1]]]
           
            # TODO: Improve margins here. Should depend on crop size.
            xmin, ymin, xmax, ymax = item_obj_1[0]['bounding_box']
            box_1 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m1 = get_box(box_1)

            xmin, ymin, xmax, ymax = item_obj_2[0]['bounding_box']
            box_2 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m2 = get_box(box_2)

            
            # print(box_1)
            # plt.imsave('m1.jpg', m1.squeeze().numpy())
            # plt.imsave('img.jpg', np.array(data_pair_file['rgb'], dtype=np.uint8))
            # # print(m1.squeeze().shape, np.array(data_pair_file['rgb']).shape)
            # rgb_image = np.array(data_pair_file['rgb'], dtype=np.uint8)
            # plt.imsave('img_crop.jpg', rgb_image[box_1[0][1]:box_1[1][1], box_1[0][0]:box_1[1][0], :])

            # time.sleep(10)
            data = {'mask_1': m1,
                    'mask_2': m2, 
                    'image': Image.fromarray(np.array(data_pair_file['rgb'], dtype=np.uint8)),
                    'is_self_feature': pos}
            self.transform(data)
        
            data_pair.append(data)

        # create dict and return
        return data_pair[0], data_pair[1]
