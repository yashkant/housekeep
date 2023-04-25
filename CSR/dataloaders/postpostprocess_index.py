#! /srv/cvmlp-lab/flash1/gchhablani3/miniconda3/envs/csr/bin/python

#SBATCH -o slurm_output_%j.txt
#SBATCH -e slurm_err_%j.txt
#SBATCH --cpus-per-task=12
#SBATCH -J copy_data
#SBATCH -p short

import os
import sys
import json
import shutil
import torch
import numpy as np
from PIL import Image
import tqdm

from datetime import datetime

# move the path to CSR (parent dir)
sys.path.insert(0, os.path.dirname(os.getcwd()))
from shared.utils import get_box
import dataloaders.augmentations as A
from models.backbones import FeatureLearner
from shared.constants import (COLOR_JITTER_BRIGHTNESS,
                                  COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                                  COLOR_JITTER_SATURATION,
                                  GRAYSCALE_PROBABILITY, IMAGE_SIZE, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)

import multiprocessing

def compute_resnet_features(resnet_backbone, data):

    assert torch.is_tensor(data['image'])
    assert data['image'].size()[0] == 3, \
        'actual dimensions of image are {}'.format(data['image'])

    tensor_input = torch.cat([data['image'], data['mask_1'], data['mask_2']],
                                dim=0)
    assert tensor_input.size()[0] == 5 

    tensor_input = tensor_input.unsqueeze(0)

    return resnet_backbone(tensor_input)


def get_mask_from_bb(item_obj):

    xmin, ymin, xmax, ymax = item_obj['bounding_box']
    box = np.array([[max(xmin-5, 0), max(ymin-5, 0)], 
                    [min(xmax+5+1, 255), min(ymax+5+1, 255)]
                    ]) # TODO: Improve margins here. Should depend on crop size.

    return get_box(box)

def logP(st):
    pass
    print(st)
    logfile.write(st)

def thread(o_tuple):

    o1, o2 = o_tuple
    logP(f'processing {o1}, {o2}\n')

    files = np.argwhere(index['arr'][o1,o2,:] == 1).reshape(-1)

    if len(files) == 0: return 0

    for i, fidx in enumerate(files):

        fp = index['files'][fidx] # consistently using fp when saving RGB/resnet features  

        data_dict = file_dict(fp)
        item_obj_1 = [item for item in data_dict['items'] if item['iid']==index['iids'][o1]][0]
        item_obj_2 = [item for item in data_dict['items'] if item['iid']==index['iids'][o2]][0]

        index_new[o1][o2].append((fidx,item_obj_1['bounding_box'],item_obj_2['bounding_box']))

        os.makedirs(os.path.join(target_dir, fp.split('|')[0], 
                                    'baseline_phasic_oracle','images'), 
                        exist_ok=True) # images folder
        image_filepath = os.path.join(target_dir, fp.split('|')[0], 
                                        'baseline_phasic_oracle','images',
                                        fp.split('|')[1]).replace('.json','.png') # image filepath
        # saving image
        Image.fromarray(np.array(data_dict['rgb'], dtype=np.uint8)).save(image_filepath)
        logP(f'{o1}, {o2}: image saved to {image_filepath}\n')

        # compute mask and transformed image
        data = {'mask_1': get_mask_from_bb(item_obj_1),
                'mask_2': get_mask_from_bb(item_obj_2), 
                'image': Image.fromarray(np.array(data_dict['rgb'], dtype=np.uint8)),
                'is_self_feature': o1==o2, #TODO: verify
                }
        A.TestTransform(data) # [resize masks + image, normalize image]

        # pass image and mask pair through resnet
        resnet_output = compute_resnet_features(resnet_backbone, data)

        os.makedirs(os.path.join(target_dir, fp.split('|')[0], 
                                    'baseline_phasic_oracle','resnet'), 
                        exist_ok=True) # resnet vectors folder
        resnet_filepath = os.path.join(target_dir, fp.split('|')[0], 
                                        'baseline_phasic_oracle','resnet',
                                        '{}_{}_{}'.format(o1, o2, fp.split('|')[1].replace('.json','.pt'))
                                        )# resnet vector filepath
        logP(f'{o1}, {o2}: resnet saved to {resnet_filepath}\n')

        # saving resnet vector as torch
        torch.save(resnet_output, resnet_filepath)

    indices_partwise_path = os.path.join(target_dir, 'indices_partwise')

    os.makedirs(indices_partwise_path, exist_ok=True) # scene indices partwise path folder
    torch.save({'arr':index_new[o1][o2]}, os.path.join(indices_partwise_path, '{}_{}_indices.pt'.format(o1, o2)))
    logP('{}, {}: indices saved to {}\n'.format(o1, o2, os.path.join(indices_partwise_path, '{}_{}_indices.pt'.format(o1, o2))))

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%m-%Y_%H-%M-%S")

logfile = open(f'logfile_{timestampStr}.log', 'w')

target_dir = f'/srv/rail-lab/flash5/kvr6/dev/data/csr_full_{timestampStr}' # target dir (CHANGE FULL/MINI BASED ON NUM OF OBJECTS)
shutil.rmtree(target_dir, ignore_errors=True)
os.makedirs(target_dir, exist_ok=True)

root_dir = '/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw'
logP(f"Loading index from {os.path.join(root_dir, '..', 'all_scenes_indices.pt')}")
index = torch.load(os.path.join(root_dir, '..', 'all_scenes_indices.pt'))
file_dict = lambda fp: json.load(open(os.path.join(root_dir, fp.split('|')[0], 
                                                'baseline_phasic_oracle',
                                                'csr',
                                                fp.split('|')[1])))

index_new = [[[] for _ in index['iids']] for _ in index['iids']]

# initialize resnet network
resnet_backbone = FeatureLearner(
                    in_channels=5,
                    channel_width=64,
                    pretrained=True,
                    num_classes=0,
                    backbone_str='resnet18')

pool = multiprocessing.Pool(12)

old_dir = '/srv/rail-lab/flash5/kvr6/dev/data/csr_full_23-04-2023_11-12-35'

logP("Starting conversion.....")
jobs = []
for o1 in range(len(index['iids'])): #range(5)
    for o2 in range(len(index['iids'])): #range(5)

        check_path = os.path.join(old_dir, 'indices_partwise', f'{o1}_{o2}_indices.pt')

        if os.path.exists(check_path):
            logP(f'{o1}, {o2} already processed, skipping')
            continue

        else:
            jobs.append((o1, o2))

logP('Jobs start now!')

for _ in tqdm.tqdm(pool.map(thread, jobs), total=len(jobs)):
    pass

logP(f"Saving index to {os.path.join(target_dir, 'all_scenes_indices.pt')}")
torch.save({'files':index['files'], 'iids':index['iids']}, os.path.join(target_dir, 'all_scenes_indices.pt'))