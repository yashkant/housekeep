import os
import json
import numpy as np
import torch
from tqdm import tqdm

csr_dir = os.getcwd()
# scenes_all = os.listdir('./csr_raw') # csr_raw/[scene]/[algorithm]
scenes_all = ['ihlen_1_int']
iid_frame_dict = {}
iids = []

files_all = []

for scene in scenes_all:

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
}, 'all_scenes_indices.pt')
