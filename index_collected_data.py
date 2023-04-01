import os
import json
import numpy as np

csr_dir = os.pwd()
scenes_all = os.listdir('./csr_raw') # csr_raw/[scene]/[algorithm]

iid_frame_dict = {}
iids = []

files_all = []

for scene in scenes_all:

    files_scene = os.listdir(f'./csr_raw/{scene}/algoalgo')

    files_all += [f'{scene}|{fil}' for fil in files_scene]

    for fil in files_scene:
        path = os.path.join(csr_dir, f'csr_raw/{scene}/algoalgo', files_scene[0])

        with open(path, 'r') as f:
            data = json.load(f)

        path = os.path.join(csr_dir, fil)
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