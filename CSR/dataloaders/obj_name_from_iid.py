import os
import json
from itertools import chain
from copy import deepcopy

import torch

scene = 'ihlen_0_int'
source_dir = f'/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/{scene}/baseline_phasic_oracle/csr'
target_dir = f'/srv/rail-lab/flash5/kvr6/dev/csr_full_final'

assert os.path.exists(source_dir), f'Path does not exist: {source_dir}'

iid2objname = dict({})
iid2objkey = dict({})

episode_object_mapping = dict({})

inconsistent_iids = []

for file in os.listdir(source_dir):

    # print(f'Processing {file}...')

    with open(os.path.join(source_dir, file), 'r') as fh:
        data = json.load(fh) # keys: ['rgb', 'mask', 'depth', 'items', 'current_mapping', 'correct_mapping']

        # '''     
        # Task 1: Map iid (instance ids) to object/receptacle name (Note: multiple iids can point to one object class)
        for item in data['items']: # keys: ['iid', 'sim_id', 'obj_key', 'type', 'bounding_box', 'cropped_image']
            iid = item['iid']
            objkey = item['obj_key']

            # save iid to iid2objkey
            if iid in iid2objkey.keys():
                if not iid2objkey[iid] == objkey:
                    if iid not in inconsistent_iids: inconsistent_iids.append(iid)
                    print(f'{iid} | new: {objkey} and saved:{iid2objkey[iid]} are inconsistent!')

            else:
                iid2objkey[iid] = objkey

            if item['type'] == 'rec': # receptacle: storage_room_0-table_14_0.urdf
                room_recep_compound = objkey.split('.')[0]

                room_indexed, recep_indexed = room_recep_compound.split('-')

                room_name_split = [k for k in room_indexed.split('_') if not k.isdigit()] # [storage, room]
                recep_name_split = [k for k in recep_indexed.split('_') if not k.isdigit()] # [table]

                final_name = '_'.join(room_name_split) + '|' + '_'.join(recep_name_split)

            elif item['type'] == 'obj': # object: condiment_1

                obj_name_split = [k for k in objkey.split('_') if not k.isdigit()]

                final_name = '_'.join(obj_name_split)

            else:
                raise KeyError('{} does not belong'.format(item['type']))

            # save iid to iid2objname
            if iid in iid2objname.keys():
                if not iid2objname[iid] == final_name:
                    if iid not in inconsistent_iids: inconsistent_iids.append(iid)
                    print(f'{iid} | new: {final_name} and saved:{iid2objname[iid]} are inconsistent!')

            else:
                iid2objname[iid] = final_name
        # '''

        # Task 2: Map episode name to current mapping and correct mapping
        #TODO: once iid issue is solved, save iid to iid mapping, not obj_key to obj_key mapping
        episode_number = file.split('.')[0].split('_')[1]
        assert episode_number.isdigit()

        episode_object_mapping[f'{scene}|{episode_number}'] = dict({
                'current_mapping': deepcopy(data['current_mapping']),
                'correct_mapping': deepcopy(data['correct_mapping'])
                })

torch.save(dict({'iid_to_obj_key': iid2objkey,    
                   'iid_to_obj_name': iid2objname,
                   'scene_episode_to_object_mapping': episode_object_mapping,
                   'inconsistent_iids': inconsistent_iids}), os.path.join(target_dir, 'master_iid_mapping.pkl'))


# for k, v in iid2objname.items(): print(k, v)
# input('wait')

