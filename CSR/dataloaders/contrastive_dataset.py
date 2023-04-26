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
# 106 objects and 33 receptacles = 139 total
object_key_filter = ['dutch_oven_1', 'chocolate_milk_pods_1', 'chocolate_1', 'antidepressant_1', 'wipe_warmer_1', 'sponge_1', 'herring_fillets_1', 'dumbbell_rack_1', 'thermal_laminator_1', 'camera_1', 'soap_dish_1', 'teapot_1', 'bleach_cleanser_1', 'bath_sheet_1', 'gloves_1', 'towel_1', 'string_lights_1', 'skillet_lid_1', 'knife_block_1', 'spoon_rest_1', 'lamp_1', 'fork_1', 'mini_soccer_ball_1', 'salt_shaker_1', 'sushi_mat_1', 'helmet_1', 'condiment_1', 'mustard_bottle_1', 'set-top_box_1', 'weight_loss_guide_1', 'cracker_box_1', 'skillet_1', 'toothbrush_pack_1', 'medicine_1', 'lime_squeezer_1', 'tomato_soup_can_1', 'water_bottle_1', 'candle_holder_1', 'cereal_1', 'diaper_pack_1', 'flashlight_1', 'baseball_1', 'clock_1', 'toaster_1', 'heavy_master_chef_can_1', 'spoon_1', 'handbag_1', 'ramekin_1', 'golf_ball_1', 'umbrella_1', 'plant_saucer_1', 'vase_1', 'fruit_snack_1', 'cake_mix_1', 'dishtowel_1', 'hair_dryer_1', 'dustpan_and_brush_1', 'peppermint_1', 'tea_pods_1', 'saute_pan_1', 'sparkling_water_1', 'chopping_board_1', 'coffee_pods_1', 'candy_bar_1', 'softball_1', 'can_opener_1', 'hat_1', 'lantern_1', 'utensil_holder_1', 'master_chef_can_1', 'portable_speaker_1', 'tablet_1', 'electric_heater_1', 'table_lamp_1', 'candy_1', 'coffee_beans_1', 'light_bulb_1', 'dietary_supplement_1', 'cloth_1', 'sanitary_pads_1', 'spatula_1', 'pitcher_base_1', 'washcloth_1', 'plate_1', 'potted_meat_can_1', 'saucer_1', 'blender_jar_1', 'dish_drainer_1', 'gelatin_box_1', 'soap_dispenser_1', 'fondant_1', 'racquetball_1', 'dumbbell_1', 'coffeemaker_1', 'incontinence_pads_1', 'pan_1', 'laptop_1', 'router_1', 'tennis_ball_1', 'plant_1', 'chocolate_box_1', 'shredder_1', 'electric_toothbrush_1', 'dispensing_closure_1', 'sponge_dish_1', 'tampons_1',
                     'bathroom_0-sink_49_0.urdf', 'kitchen_0-counter_80_0.urdf', 'dining_room_0-table_19_0.urdf', 'bedroom_1-chest_12_0.urdf', 'kitchen_0-counter_79_0.urdf', 'living_room_0-coffee_table_26_0.urdf', 'bedroom_0-bed_33_2.urdf', 'bedroom_1-carpet_41_0.urdf', 'dining_room_0-chair_20_0.urdf', 'kitchen_0-top_cabinet_63_0.urdf', 'kitchen_0-counter_65_0.urdf', 'kitchen_0-top_cabinet_62_0.urdf', 'kitchen_0-counter_64_0.urdf', 'dining_room_0-carpet_52_0.urdf', 'bedroom_2-chair_11_0.urdf', 'bedroom_0-bottom_cabinet_1_0.urdf', 'bedroom_1-table_14_0.urdf', 'bathroom_0-bathtub_46_0.urdf', 'kitchen_0-sink_77_0.urdf', 'bathroom_0-toilet_47_0.urdf', 'kitchen_0-bottom_cabinet_66_0.urdf', 'living_room_0-bottom_cabinet_28_0.urdf', 'bedroom_1-chest_13_0.urdf', 'kitchen_0-fridge_61_0.urdf', 'corridor_0-carpet_51_0.urdf', 'kitchen_0-shelf_83_0.urdf', 'kitchen_0-dishwasher_76_0.urdf', 'living_room_0-sofa_chair_24_0.urdf', 'bedroom_0-bottom_cabinet_0_0.urdf', 'bedroom_2-bed_37_2.urdf', 'living_room_0-carpet_53_0.urdf', 'kitchen_0-bottom_cabinet_no_top_70_0.urdf', 'kitchen_0-oven_68_0.urdf'] 

def obj_episode_filters(data_split, test_unseen_objects = True):
    if data_split == DataSplit.TRAIN:
        use_obj = lambda o: (0 <= o < 92) or o > 105
        use_episode = lambda e: e > 10
    elif data_split == DataSplit.VAL:
        use_obj = lambda o: (90 <= o < 95) or o > 105
        use_episode = lambda e: e > 10
    elif data_split == DataSplit.TEST:
        if test_unseen_objects:
            use_obj = lambda o: (95 <= o < 106) or o > 105
        else:
            print('Using ALL objects for testing')
            use_obj = lambda o: True
        use_episode = lambda e: e <= 10
    else:
        assert False, 'Data split not recognized'
    return use_obj, use_episode

class ContrastiveDataset(Dataset):

    def __init__(self, root_dir, transform, data_split: DataSplit, test_unseen_objects = True):
        # set the root directory
        self.root_dir = root_dir
        self.data_split = data_split

        use_obj, use_episode = obj_episode_filters(data_split, test_unseen_objects)
        
        # ## USE ALL OBJECTS AND EPISODES
        # use_obj = lambda o: True
        # use_episode = lambda o: True

        self.resnet_path = os.path.join(root_dir, 'ihlen_1_int', 
                                    'baseline_phasic_oracle','resnet'
                                    )

        self.scene_indices_array = [[[] for _ in object_key_filter] for _ in object_key_filter]

        for file in os.listdir(self.resnet_path):
            o1, o2, _, episodestep = file.split('_')
            o1 = int(o1)
            o2 = int(o2)
            episodestep = int(episodestep.replace('.pt',''))
            episode = episodestep//1000
            if use_episode(episode):
                if use_obj(o1) and use_obj(o2):
                    if o1 > 105 and o2 > 105:
                        if random.random() < 0.5: continue
                    self.scene_indices_array[o1][o2].append((file,episode))

        print('Scenes Indices Array created!!')

        # num_good_pairs = 0
        # num_bad_pairs = 0

        # for o1 in range(len(object_key_filter)):
        #     for o2 in range(len(object_key_filter)):
        #         if o1 < 106 or o2 < 106:
        #             num_good_pairs += len(self.scene_indices_array[o1][o2])
        #         else:
        #             num_bad_pairs += len(self.scene_indices_array[o1][o2])

        # print(num_good_pairs, num_bad_pairs)

        self.indexed_data = []
        for o1 in range(len(object_key_filter)):
            for o2 in range(len(object_key_filter)):
                files_and_episodes = self.scene_indices_array[o1][o2]
                for i,(f1,e1) in enumerate(files_and_episodes):
                    assert use_obj(o1) and use_obj(o2)
                    for (f2,e2) in files_and_episodes[i+1:]:
                        if e1 == e2: # positive pairs must come from the same scene and episode
                            self.indexed_data.append((o1, o2, f1, f2))

        print('Indexed data created!!')

        self.length = len(self.indexed_data)
        print('Total length of dataset type ', data_split, '::: ', self.length)

        # # save the data augmentations that are to be applied to the images
        # self.transform = transform
        # assert self.transform is not None
        self.on_episode = -1

    def get_next_episode(self):
        self.on_episode += 1
        if self.on_episode >= 10:
            return None, None
        episode_resnets = [[[] for _ in object_key_filter] for _ in object_key_filter]
        for o1 in range(len(object_key_filter)):
            for o2 in range(len(object_key_filter)):
                for file, episode in self.scene_indices_array[o1][o2]:
                    if episode == self.on_episode:
                        full_file_path = os.path.join(self.resnet_path, file)
                        tensor_data = torch.load(full_file_path)
                        episode_resnets[o1][o2].append(tensor_data)
        return episode_resnets, self.on_episode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obj_1, obj_2, file_path_1, file_path_2 = self.indexed_data[idx]

        data_pair = []
        for file_path in [file_path_1, file_path_2]:
            full_file_path = os.path.join(self.resnet_path, file_path)
            tensor_data = torch.load(full_file_path)
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