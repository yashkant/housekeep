import os
import json
import numpy as np
import torch
from tqdm import tqdm

csr_dir = os.getcwd()
# scenes_all = os.listdir('./csr_raw') # csr_raw/[scene]/[algorithm]

# merom_1_int 4766 4766
# beechwood_0_int 6998 6998
# pomaria_0_int 4034 4034
# beechwood_1_int 6169 6169
# benevolence_2_int 5753 5753
# merom_0_int 3915 3915
# rs_int 4905 4905
# ihlen_1_int 4035 4035
# pomaria_2_int 4413 4413
# wainscott_0_int 4793 4793
# benevolence_1_int 6294 6294

# scenes:
#   test:
#   - data/scene_datasets/igibson/scenes/Benevolence_1_int.glb
#   - data/scene_datasets/igibson/scenes/Ihlen_0_int.glb
#   - data/scene_datasets/igibson/scenes/Beechwood_1_int.glb
#   - data/scene_datasets/igibson/scenes/Merom_0_int.glb
#   train:
#   - data/scene_datasets/igibson/scenes/Benevolence_2_int.glb
#   - data/scene_datasets/igibson/scenes/Wainscott_0_int.glb
#   - data/scene_datasets/igibson/scenes/Beechwood_0_int.glb
#   - data/scene_datasets/igibson/scenes/Rs_int.glb
#   - data/scene_datasets/igibson/scenes/Pomaria_1_int.glb
#   - data/scene_datasets/igibson/scenes/Pomaria_0_int.glb
#   - data/scene_datasets/igibson/scenes/Pomaria_2_int.glb
#   - data/scene_datasets/igibson/scenes/Wainscott_1_int.glb
#   val:
#   - data/scene_datasets/igibson/scenes/Ihlen_1_int.glb
#   - data/scene_datasets/igibson/scenes/Merom_1_int.glb

train_scenes = ['beechwood_0_int', 'pomaria_0_int', 'benevolence_2_int', 'rs_int', 'pomaria_2_int', 'wainscott_0_int']
val_scenes = ['merom_1_int', 'ihlen_1_int']
test_scenes = ['benevolence_1_int',  'beechwood_1_int', 'merom_0_int']

scene_files = ['train_indices.pt', 'val_indices.pt', 'test_indices.pt']
iid_frame_dict = {}
iids = []

files_all = []

for scenes, scene_file in zip([train_scenes, val_scenes, test_scenes], scene_files):
    for scene in scenes:
        files_scene = os.listdir(f'./csr_raw/{scene}/baseline_phasic_oracle/csr')

        files_all += [f'{scene}|{fil}' for fil in files_scene]

        for fil in tqdm(files_scene):
            path = os.path.join(f'./csr_raw/{scene}/baseline_phasic_oracle/csr', fil)
            with open(path) as f:
                data = json.load(f)

            for item in data['items']:

                if item['iid'] not in iids:
                    iids.append(item['iid'])

                if item['iid'] not in iid_frame_dict:
                    iid_frame_dict[item['iid']] = {}

                for item_2 in data['items']:

                    if item_2['iid'] not in iid_frame_dict[item['iid']]:
                        iid_frame_dict[item['iid']][item_2['iid']] = [f'{scene}|{path}']

                    else:
                        iid_frame_dict[item['iid']][item_2['iid']].append(f'{scene}|{path}')

    arr = np.zeros((len(iids), len(iids), len(files_all)))

    for iid in iid_frame_dict.keys():
        for iid_2 in iid_frame_dict[iid].keys():
            for fil in iid_frame_dict[iid][iid_2]:

                short_file_name = '{}|{}'.format(fil.split('|')[0], fil.split('/')[-1])
                arr[iids.index(iid), iids.index(iid_2), files_all.index(short_file_name)] = 1

    # for iid in iid_frame_dict.keys():
    #     print(len(iid_frame_dict[iid][iid]))

    # for iid in iid_frame_dict.keys():
    #     s = 0
    #     for iid_2 in iid_frame_dict[iid].keys():
    #         if iid != iid_2:
    #             s += len(iid_frame_dict[iid][iid_2])
    #     print(s)

    torch.save({
        'iids': iids,
        'files': files_all,
        'arr': arr
    }, scene_file)
