import os
import sys
import json

import numpy as np
import random
from PIL import Image

import torch
import torch.nn.functional as F
# from transformers import CLIPImageProcessor
from transformers import CLIPProcessor, CLIPModel
from torchvision.io import read_image

sys.path.append('/srv/rail-lab/flash5/mpatel377/dev/housekeep_csr/CSR')

from shared.data_split import DataSplit
from lightning.modules.moco2_module_mini import MocoV2Lite

# ---- CONSTANTS
OBJECT_KEY_FILTER = ['dutch_oven_1', 'chocolate_milk_pods_1', 'chocolate_1', 'antidepressant_1', 'wipe_warmer_1', 'sponge_1', 'herring_fillets_1', 'dumbbell_rack_1', 'thermal_laminator_1', 'camera_1', 'soap_dish_1', 'teapot_1', 'bleach_cleanser_1', 'bath_sheet_1', 'gloves_1', 'towel_1', 'string_lights_1', 'skillet_lid_1', 'knife_block_1', 'spoon_rest_1', 'lamp_1', 'fork_1', 'mini_soccer_ball_1', 'salt_shaker_1', 'sushi_mat_1', 'helmet_1', 'condiment_1', 'mustard_bottle_1', 'set-top_box_1', 'weight_loss_guide_1', 'cracker_box_1', 'skillet_1', 'toothbrush_pack_1', 'medicine_1', 'lime_squeezer_1', 'tomato_soup_can_1', 'water_bottle_1', 'candle_holder_1', 'cereal_1', 'diaper_pack_1', 'flashlight_1', 'baseball_1', 'clock_1', 'toaster_1', 'heavy_master_chef_can_1', 'spoon_1', 'handbag_1', 'ramekin_1', 'golf_ball_1', 'umbrella_1', 'plant_saucer_1', 'vase_1', 'fruit_snack_1', 'cake_mix_1', 'dishtowel_1', 'hair_dryer_1', 'dustpan_and_brush_1', 'peppermint_1', 'tea_pods_1', 'saute_pan_1', 'sparkling_water_1', 'chopping_board_1', 'coffee_pods_1', 'candy_bar_1', 'softball_1', 'can_opener_1', 'hat_1', 'lantern_1', 'utensil_holder_1', 'master_chef_can_1', 'portable_speaker_1', 'tablet_1', 'electric_heater_1', 'table_lamp_1', 'candy_1', 'coffee_beans_1', 'light_bulb_1', 'dietary_supplement_1', 'cloth_1', 'sanitary_pads_1', 'spatula_1', 'pitcher_base_1', 'washcloth_1', 'plate_1', 'potted_meat_can_1', 'saucer_1', 'blender_jar_1', 'dish_drainer_1', 'gelatin_box_1', 'soap_dispenser_1', 'fondant_1', 'racquetball_1', 'dumbbell_1', 'coffeemaker_1', 'incontinence_pads_1', 'pan_1', 'laptop_1', 'router_1', 'tennis_ball_1', 'plant_1', 'chocolate_box_1', 'shredder_1', 'electric_toothbrush_1', 'dispensing_closure_1', 'sponge_dish_1', 'tampons_1',
                     'bathroom_0-sink_49_0.urdf', 'kitchen_0-counter_80_0.urdf', 'dining_room_0-table_19_0.urdf', 'bedroom_1-chest_12_0.urdf', 'kitchen_0-counter_79_0.urdf', 'living_room_0-coffee_table_26_0.urdf', 'bedroom_0-bed_33_2.urdf', 'bedroom_1-carpet_41_0.urdf', 'dining_room_0-chair_20_0.urdf', 'kitchen_0-top_cabinet_63_0.urdf', 'kitchen_0-counter_65_0.urdf', 'kitchen_0-top_cabinet_62_0.urdf', 'kitchen_0-counter_64_0.urdf', 'dining_room_0-carpet_52_0.urdf', 'bedroom_2-chair_11_0.urdf', 'bedroom_0-bottom_cabinet_1_0.urdf', 'bedroom_1-table_14_0.urdf', 'bathroom_0-bathtub_46_0.urdf', 'kitchen_0-sink_77_0.urdf', 'bathroom_0-toilet_47_0.urdf', 'kitchen_0-bottom_cabinet_66_0.urdf', 'living_room_0-bottom_cabinet_28_0.urdf', 'bedroom_1-chest_13_0.urdf', 'kitchen_0-fridge_61_0.urdf', 'corridor_0-carpet_51_0.urdf', 'kitchen_0-shelf_83_0.urdf', 'kitchen_0-dishwasher_76_0.urdf', 'living_room_0-sofa_chair_24_0.urdf', 'bedroom_0-bottom_cabinet_0_0.urdf', 'bedroom_2-bed_37_2.urdf', 'living_room_0-carpet_53_0.urdf', 'kitchen_0-bottom_cabinet_no_top_70_0.urdf', 'kitchen_0-oven_68_0.urdf'] 

GROUND_TRUTH_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'

RAW_DATA_PATH = '/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/ihlen_1_int/baseline_phasic_oracle/'

ROOMS = ['bathroom',
 'bedroom',
 'childs_room',
 'closet',
 'corridor',
 'dining_room',
 'exercise_room',
 'garage',
 'home_office',
 'kitchen',
 'living_room',
 'lobby',
 'pantry_room',
 'playroom',
 'storage_room',
 'television_room',
 'utility_room']
# ----

# ---- CHECKPOINTS
CSR_CKPT = '/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/CSR/static_checkpoints/model-epoch=188-val_loss=12.99.ckpt'

def obj_key_to_class(any_key):
    if '.urdf' in any_key: # receptacle: storage_room_0-table_14_0.urdf
        room_recep_compound = any_key.split('.')[0]

        room_indexed, recep_indexed = room_recep_compound.split('-')

        room_name_split = [k for k in room_indexed.split('_') if not k.isdigit()] # [storage, room]
        recep_name_split = [k for k in recep_indexed.split('_') if not k.isdigit()] # [table]

        final_name = '_'.join(room_name_split) + '|' + '_'.join(recep_name_split)
        return final_name

    else: # object: condiment_1

        any_name_split = [k for k in any_key.split('_') if not k.isdigit()]

        return '_'.join(any_name_split)


class GeneratePreferenceDataset():

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 user_personas_path: str,
                 image_input: bool = False,
                 max_annotations = -1,
                 test_unseen_objects = True,
                 housekeep_path = './housekeep.npy'
                 ):
     
        self.image_input = image_input

        # npy_data = np.load(housekeep_path, allow_pickle=True)

        self.rooms_hkp = ROOMS   # npy_data['rooms']

        self.user_encoding_matrix = F.one_hot(torch.arange(0, 10))

        self.room_encoding_matrix = F.one_hot(
            torch.arange(0, len(self.rooms_hkp)))

        if data_split == DataSplit.TRAIN:
            self.use_obj = lambda o: o in np.arange(0,92) or o > 105
            self.use_episode = lambda e: e > 10
        elif data_split == DataSplit.VAL:
            self.use_obj = lambda o: o in np.arange(90,95) or o > 105
            self.use_episode = lambda e: e > 10
        elif data_split == DataSplit.TEST:
            if test_unseen_objects:
                self.use_obj = lambda o: o in np.arange(95,106) or o > 105
            else:
                self.use_obj = lambda o: o in np.arange(0,95) or o > 105
            self.use_episode = lambda e: e <= 10
        else:
            assert False, 'Data split not recognized'

        self.is_obj = lambda o: o < 106

        if image_input:
            raise NotImplementedError("Image input not implemented yet")
        else:
            self.CSRprocess = MocoV2Lite().load_from_checkpoint(CSR_CKPT)

        self.CSRprocess.eval()
        self.get_csr = lambda resnet_vec: torch.nn.functional.normalize(self.CSRprocess.projection_q(resnet_vec), dim=-1).squeeze(0).detach()
        
        self.split_str = {DataSplit.TRAIN:'seen', DataSplit.TEST:'unseen-test', DataSplit.VAL:'unseen-val'}[data_split]
        
        # self.episode_run_numbers = \
        #     [x.split('_')[1].split('.')[0] 
        #      for x in os.listdir(os.path.join(root_dir, 'ihlen_1_int', 'baseline_phasic_oracle', 'images'))]

        try:
            files_list = torch.load(GROUND_TRUTH_NAMES_FILE)['files']
        except:
            files_list = torch.load(GROUND_TRUTH_NAMES_FILE.replace('coc','coc'))['files']

        self.ep_r_nums = [int(f.replace('.json','').replace('obs_','')) for f in files_list]

        self.get_resnet = lambda episode_run_num, obj_1, obj_2 : torch.load(os.path.join(root_dir, 
                                        'ihlen_1_int', 
                                        'baseline_phasic_oracle',
                                        'resnet',
                                        '{}_{}_csr_{}.pt'.format(obj_1, obj_2, episode_run_num)
                                        ))
        
        self.get_image = lambda episode_run_num : read_image(os.path.join(root_dir, 
                                        'ihlen_1_int', 
                                        'baseline_phasic_oracle',
                                        'images',
                                        'csr_{}.png'.format(episode_run_num)
                                        ))
        
        ## TODO: Verify that this is the correct way to get the CLIP features
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



        self.user_persona_dict = torch.load(user_personas_path)

        try:
            gt_edges = torch.load(GROUND_TRUTH_FILE)
        except:
            gt_edges = torch.load(GROUND_TRUTH_FILE.replace('coc','coc'))

        self.find_mapping = lambda ep_r_num, item1_name, item2_name: \
            gt_edges[self.ep_r_nums.index(ep_r_num), 
                    OBJECT_KEY_FILTER.index(item1_name), 
                    OBJECT_KEY_FILTER.index(item2_name)].to(int)

        self.root_dir = root_dir

    def get_features(self, episode_run_num, object_idx):

        # image = self.get_image(episode_run_num)

        # if self.image_input:
        #     csr_input = image

        # else:
        csr_input = self.get_resnet(episode_run_num, 
                                        object_idx, object_idx) #TODO: confirm

        csr_feature = self.get_csr(csr_input)

        bb_data_path = os.path.join(RAW_DATA_PATH,
                                    f'csr/csr_{episode_run_num}.json')

        with open(bb_data_path, 'r') as fh:
            crop_data_dict = json.load(fh)
            item_picked = [item for item in crop_data_dict['items'] 
                    if item['obj_key'] == OBJECT_KEY_FILTER[object_idx]][0]

        # obs_data_path = os.path.join(RAW_DATA_PATH, 
        #                             f'observations/obs_{episode_run_num}.json')

        # with open(obs_data_path, 'r') as fh:
        #     obs_data_dict = json.loads(json.load(fh))[0]
        #     image_full = obs_data_dict['rgb']

        # xmin, ymin, xmax, ymax = item_picked['bounding_box']
        # cropped_image = np.array(image_full)[ymin:ymax+1, xmin:xmax+1]

        cropped_image = Image.fromarray(np.array(item_picked['cropped_image'], dtype=np.uint8))

        clip_feature = None
        try:
            processed = self.processor(images=cropped_image, return_tensors="pt", padding=True)
            clip_feature = self.model.get_image_features(pixel_values=processed['pixel_values'])
        except ValueError as e:
            print(cropped_image.size)
            print("Clip failed for ",e)

        return csr_feature, clip_feature

    def get_preference(self, o1, o2, ep_run_num, persona_id=None):

        assert isinstance(o1, int) and isinstance(o2, int)

        item1_key = OBJECT_KEY_FILTER[o1]
        item1_class = obj_key_to_class(item1_key)

        item2_key = OBJECT_KEY_FILTER[o2]
        item2_class = obj_key_to_class(item2_key)

        room = item2_class.split('|')[0]

        if persona_id is None:
            raise KeyError(f'Persona id required')

        if self.find_mapping(ep_run_num, item1_key, item2_key): # episode map based on keys

            # preference mapping based on object class
            try:
                persona_preferences = \
                    self.user_persona_dict[f'persona_{persona_id}'][self.split_str]['{}/{}'.format(item1_class, room)]
            except KeyError as e:
                print(f'KeyError: {e}')
                return None

            if item2_class in persona_preferences:
                label = 1

            else:
                label = 0

        else:
            label = None # ignore these examples

        return label

    def generate_data(self):

        from datetime import datetime
        datetime_object = datetime.now()
        timestampStr = datetime_object.strftime("%d-%m-%Y_%H:%M:%S")

        resnet_path = os.path.join(self.root_dir, 'ihlen_1_int', 
                                    'baseline_phasic_oracle','resnet'
                                    )

        tensor_data = []

        for file_idx, file in enumerate(os.listdir(resnet_path)):
            o1, o2, _, episodestep = file.split('_')

            o1 = int(o1)
            o2 = int(o2)

            episodestep = int(episodestep.replace('.pt',''))
            episode = episodestep//1000

            if self.use_episode(episode):
                if self.use_obj(o1) and self.use_obj(o2):
                    if self.is_obj(o1) and not self.is_obj(o2):
                        print(file_idx, file)
                        
                        csr_feature1, clip_feature1 = self.get_features(episodestep, o1)
                        if clip_feature1 is None:
                            continue
                        csr_feature2, clip_feature2 = self.get_features(episodestep, o2)
                        if clip_feature2 is None:
                            continue

                        room_o2 = OBJECT_KEY_FILTER[o2]
                        
                        room_o2 = obj_key_to_class(room_o2).split('|')[0]

                        room_feat = self.room_encoding_matrix[self.rooms_hkp.index(room_o2), :]

                        for persona_id in range(10):
                            user_embb = self.user_encoding_matrix[persona_id, :]

                            label = self.get_preference(o1, o2, episodestep, persona_id)

                            if label is None: continue
                            print('Label exists')
                            #TODO: make a dictionary of all data required
                            tensor_data.append(dict({
                                'key_item_1': OBJECT_KEY_FILTER[o1],
                                'key_item_2': OBJECT_KEY_FILTER[o2],
                                'csr_item_1': csr_feature1,
                                'csr_item_2': csr_feature2,
                                'clip_item_1': clip_feature1,
                                'clip_item_2': clip_feature2,
                                'room_embb': room_feat,
                                'persona_id': persona_id,
                                'persona_embb': user_embb,
                                'label': label
                            })) 
                            print('here222')

            if file_idx > 0 and file_idx%200 == 0:
                torch.save(tensor_data, os.path.join('/srv/rail-lab/flash5/mpatel377/data/csr_clip_preferences', f'{self.split_str}_partial_preferences_{file_idx}.pt'))

train_generator = GeneratePreferenceDataset(                 
                 root_dir= '/srv/rail-lab/flash5/kvr6/dev/data/csr_full_v2_25-04-2023_22-06-27', 
                 data_split= DataSplit.TRAIN,
                 user_personas_path= '/srv/rail-lab/flash5/kvr6/dev/all_preferences_26-04-2023_11-48-18.pt',
                 image_input= False,
                 max_annotations = -1,
                 test_unseen_objects = True,
                 housekeep_path = '/srv/rail-lab/flash5/kvr6/dev/data/housekeep.npy'
)
print('train generator initialized')
train_generator.generate_data()

test_generator = GeneratePreferenceDataset(                 
                 root_dir= '/srv/rail-lab/flash5/kvr6/dev/data/csr_full_v2_25-04-2023_22-06-27', 
                 data_split= DataSplit.TEST,
                 user_personas_path= '/srv/rail-lab/flash5/kvr6/dev/all_preferences_26-04-2023_11-48-18.pt',
                 image_input= False,
                 max_annotations = -1,
                 test_unseen_objects = True,
                 housekeep_path = '/srv/rail-lab/flash5/kvr6/dev/data/housekeep.npy'
)
print('test generator initialized')
test_generator.generate_data()

val_generator = GeneratePreferenceDataset(                 
                 root_dir= '/srv/rail-lab/flash5/kvr6/dev/data/csr_full_v2_25-04-2023_22-06-27', 
                 data_split= DataSplit.VAL,
                 user_personas_path= '/srv/rail-lab/flash5/kvr6/dev/all_preferences_26-04-2023_11-48-18.pt',
                 image_input= False,
                 max_annotations = -1,
                 test_unseen_objects = True,
                 housekeep_path = '/srv/rail-lab/flash5/kvr6/dev/data/housekeep.npy'
)
print('val generator initialized')
val_generator.generate_data()
