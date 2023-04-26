import os
import json
from itertools import chain
from copy import deepcopy

import torch

scene = 'ihlen_1_int'
assert os.path.exists(f'/srv/rail-lab/flash5/kvr6/dev/csr_full_final/{scene}')
source_dir = f'/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/{scene}/baseline_phasic_oracle/observations'
target_dir = f'/srv/rail-lab/flash5/kvr6/dev/csr_full_final'

assert os.path.exists(source_dir), f'Path does not exist: {source_dir}'

iid2objname = dict({})
iid2objkey = dict({})

episode_object_mapping = dict({})

inconsistent_iids = []

for file in os.listdir(source_dir):

    print(f'Processing {file}...')

    with open(os.path.join(source_dir, file), 'r') as fh:
        json_data = json.loads(json.load(fh)) # load observation file 
        cos_eor_data = json_data[0]['cos_eor']

        def process_iid(any_iid):

            any_sim_obj_id = cos_eor_data['iid_to_sim_obj_id'][str(any_iid)]
            any_key = cos_eor_data['sim_obj_id_to_obj_key'][str(any_sim_obj_id)]

            # save iid to iid2objkey
            if any_iid in iid2objkey.keys():
                if not iid2objkey[any_iid] == any_key:
                    if any_iid not in inconsistent_iids: inconsistent_iids.append(any_iid)
                    print(f'{any_iid} | new: {any_key} and saved:{iid2objkey[any_iid]} are inconsistent!')

            else:
                iid2objkey[any_iid] = any_key

            if '.urdf' in any_key: # receptacle: storage_room_0-table_14_0.urdf
                room_recep_compound = any_key.split('.')[0]

                room_indexed, recep_indexed = room_recep_compound.split('-')

                room_name_split = [k for k in room_indexed.split('_') if not k.isdigit()] # [storage, room]
                recep_name_split = [k for k in recep_indexed.split('_') if not k.isdigit()] # [table]

                final_name = '_'.join(room_name_split) + '|' + '_'.join(recep_name_split)

            else: # object: condiment_1

                any_name_split = [k for k in any_key.split('_') if not k.isdigit()]

                final_name = '_'.join(any_name_split)

            # save any_iid to iid2objname
            if any_iid in iid2objname.keys():
                if not iid2objname[any_iid] == final_name:
                    if any_iid not in inconsistent_iids: inconsistent_iids.append(any_iid)
                    print(f'{any_iid} | new: {final_name} and saved:{iid2objname[any_iid]} are inconsistent!')

            else:
                iid2objname[any_iid] = final_name

        visible_obj_iids = json_data[0]['visible_obj_iids']
        visible_rec_iids = json_data[0]['visible_recept_iids']

        for iid_list in [visible_obj_iids, visible_rec_iids]:
            for any_iid in iid_list:
                process_iid(any_iid)


# torch.save(dict({'iid_to_obj_key': iid2objkey,    
#                    'iid_to_obj_name': iid2objname,
#                    'inconsistent_iids': inconsistent_iids}), os.path.join(target_dir, 'master_iid_mapping.pkl'))
