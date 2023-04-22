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
        self.scenes_indices = torch.load(os.path.join(root_dir, '..', 'all_scenes_indices.pt'))
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
                files = np.argwhere(self.masked_scene_indices_arr[o1,o2,:] == 1).reshape(-1)
                for i,f1 in enumerate(files):
                    for f2 in files[i+1:]:
                        self.indexed_data.append((o1, o2, f1, f2))
       
        self.length = len(self.indexed_data)

        # save the data augmentations that are to be applied to the images
        self.transform = transform
        assert self.transform is not None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obj_1, obj_2, file_path_1, file_path_2 = self.indexed_data[idx]

        file_dict = lambda fp: json.load(open(os.path.join(self.root_dir, 
                                                        fp.split('|')[0], 
                                                        'baseline_phasic_oracle',
                                                        'csr',
                                                        fp.split('|')[1])))

        data_pair = []
        for file_path_idx in [file_path_1, file_path_2]:
            data_dict = file_dict(self.scenes_indices['files'][file_path_idx])
            item_obj_1 = [item for item in data_dict['items'] if item['iid']==self.current_iids[obj_1]]
            item_obj_2 = [item for item in data_dict['items'] if item['iid']==self.current_iids[obj_2]]
           
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
                    'image': Image.fromarray(np.array(data_dict['rgb'], dtype=np.uint8)),
                    'is_self_feature': obj_1==obj_2,
                    }
            self.transform(data)
        
            data_pair.append(data)

        # create dict and return
        return data_pair[0], data_pair[1]
