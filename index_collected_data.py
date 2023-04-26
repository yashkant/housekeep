import os, sys
import json
import numpy as np
import torch
import shutil
from tqdm import tqdm
from PIL import Image
import random

from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%m-%Y_%H-%M-%S")

sys.path.insert(0, './CSR')
from CSR.models.backbones import FeatureLearner
import CSR.dataloaders.augmentations as A
from CSR.shared.constants import (COLOR_JITTER_BRIGHTNESS,
                                  COLOR_JITTER_CONTRAST, COLOR_JITTER_HUE,
                                  COLOR_JITTER_SATURATION,
                                  GRAYSCALE_PROBABILITY, IMAGE_SIZE, NORMALIZE_RGB_MEAN,
                                  NORMALIZE_RGB_STD)

IMAGE_SIZE = 512

def get_box(corners, random_box=False):
    if random_box and corners is None:
        t_min, t_max = random.randint(IMAGE_SIZE), random.randint(IMAGE_SIZE)
        x_min, x_max = min(t_min, t_max), max(t_min, t_max)
        t_min, t_max = random.randint(IMAGE_SIZE), random.randint(IMAGE_SIZE)
        y_min, y_max = min(t_min, t_max), max(t_min, t_max)

        corners = [[x_min, x_max], [y_min, y_max]]

    box = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    box[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]] = 1.

    return box.unsqueeze(0)


def compute_resnet_features(resnet_backbone, input_dict):

    assert torch.is_tensor(input_dict['image'])
    assert input_dict['image'].size()[0] == 3, \
        'actual dimensions of image are {}'.format(input_dict['image'])

    tensor_input = torch.cat([input_dict['image'], input_dict['mask_1'], input_dict['mask_2']],
                                dim=0)
    assert tensor_input.size()[0] == 5 

    tensor_input = tensor_input.unsqueeze(0)

    return resnet_backbone(tensor_input)


def get_mask_from_bb(bb):

    xmin, ymin, xmax, ymax = bb
    box = np.array([[max(xmin-5, 0), max(ymin-5, 0)], 
                    [min(xmax+5+1, 255), min(ymax+5+1, 255)]
                    ]) # TODO: Improve margins here. Should depend on crop size.

    return get_box(box)


def save_image_resnet(image_np, resnet_output, obj_key_x, obj_key_y, file_path_xy):

    # image
    os.makedirs(os.path.join(target_dir, file_path_xy.split('|')[0], 
                                'baseline_phasic_oracle','images'), 
                    exist_ok=True) # images folder
    image_filepath = os.path.join(target_dir, file_path_xy.split('|')[0], 
                                    'baseline_phasic_oracle','images',
                                    file_path_xy.split('|')[1]).replace('.json','.png')# image filepath
    Image.fromarray(image_np).save(image_filepath)

    index_obj_x = object_key_filter.index(obj_key_x)
    index_obj_y = object_key_filter.index(obj_key_y)

    # resnet
    os.makedirs(os.path.join(target_dir, file_path_xy.split('|')[0], 
                                'baseline_phasic_oracle','resnet'), 
                    exist_ok=True) # resnet vectors folder
    resnet_filepath = os.path.join(target_dir, file_path_xy.split('|')[0], 
                                    'baseline_phasic_oracle','resnet',
                                    '{}_{}_{}'.format(index_obj_x, index_obj_y, 
                                                   file_path_xy.split('|')[1].replace('.json','.pt'))
                                    )# resnet vector filepath
    torch.save(resnet_output, resnet_filepath)

    return image_filepath, resnet_filepath


# ---- CONSTANTS ----

scene = 'ihlen_1_int'
root_dir = '/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw'
target_dir = f'/coc/flash5/kvr6/dev/data/csr_full_v2_{timestampStr}' # target dir (CHANGE FULL/MINI BASED ON NUM OF OBJECTS)
files_scene = \
    os.listdir(f'/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/{scene}/baseline_phasic_oracle/csr')

# ---- CONSTANTS ----

# make target dir
shutil.rmtree(target_dir, ignore_errors=True)
os.makedirs(target_dir, exist_ok=True)

files_scene = files_scene

entity_keys = [] # list of unique entity keys
key_to_frame_dict = {} # list of files for each entity key

files_all = [f'{scene}|{fil}' for fil in files_scene]

object_key_filter = ['dutch_oven_1', 'chocolate_milk_pods_1', 'chocolate_1', 'antidepressant_1', 'wipe_warmer_1', 'sponge_1', 'herring_fillets_1', 'dumbbell_rack_1', 'thermal_laminator_1', 'camera_1', 'soap_dish_1', 'teapot_1', 'bleach_cleanser_1', 'bath_sheet_1', 'gloves_1', 'towel_1', 'string_lights_1', 'skillet_lid_1', 'knife_block_1', 'spoon_rest_1', 'lamp_1', 'fork_1', 'mini_soccer_ball_1', 'salt_shaker_1', 'sushi_mat_1', 'helmet_1', 'condiment_1', 'mustard_bottle_1', 'set-top_box_1', 'weight_loss_guide_1', 'cracker_box_1', 'skillet_1', 'toothbrush_pack_1', 'medicine_1', 'lime_squeezer_1', 'tomato_soup_can_1', 'water_bottle_1', 'candle_holder_1', 'cereal_1', 'diaper_pack_1', 'flashlight_1', 'baseball_1', 'clock_1', 'toaster_1', 'heavy_master_chef_can_1', 'spoon_1', 'handbag_1', 'ramekin_1', 'golf_ball_1', 'umbrella_1', 'plant_saucer_1', 'vase_1', 'fruit_snack_1', 'cake_mix_1', 'dishtowel_1', 'hair_dryer_1', 'dustpan_and_brush_1', 'peppermint_1', 'tea_pods_1', 'saute_pan_1', 'sparkling_water_1', 'chopping_board_1', 'coffee_pods_1', 'candy_bar_1', 'softball_1', 'can_opener_1', 'hat_1', 'lantern_1', 'utensil_holder_1', 'master_chef_can_1', 'portable_speaker_1', 'tablet_1', 'electric_heater_1', 'table_lamp_1', 'candy_1', 'coffee_beans_1', 'light_bulb_1', 'dietary_supplement_1', 'cloth_1', 'sanitary_pads_1', 'spatula_1', 'pitcher_base_1', 'washcloth_1', 'plate_1', 'potted_meat_can_1', 'saucer_1', 'blender_jar_1', 'dish_drainer_1', 'gelatin_box_1', 'soap_dispenser_1', 'fondant_1', 'racquetball_1', 'dumbbell_1', 'coffeemaker_1', 'incontinence_pads_1', 'pan_1', 'laptop_1', 'router_1', 'tennis_ball_1', 'plant_1', 'chocolate_box_1', 'shredder_1', 'electric_toothbrush_1', 'dispensing_closure_1', 'sponge_dish_1', 'tampons_1',
                     'bathroom_0-sink_49_0.urdf', 'kitchen_0-counter_80_0.urdf', 'dining_room_0-table_19_0.urdf', 'bedroom_1-chest_12_0.urdf', 'kitchen_0-counter_79_0.urdf', 'living_room_0-coffee_table_26_0.urdf', 'bedroom_0-bed_33_2.urdf', 'bedroom_1-carpet_41_0.urdf', 'dining_room_0-chair_20_0.urdf', 'kitchen_0-top_cabinet_63_0.urdf', 'kitchen_0-counter_65_0.urdf', 'kitchen_0-top_cabinet_62_0.urdf', 'kitchen_0-counter_64_0.urdf', 'dining_room_0-carpet_52_0.urdf', 'bedroom_2-chair_11_0.urdf', 'bedroom_0-bottom_cabinet_1_0.urdf', 'bedroom_1-table_14_0.urdf', 'bathroom_0-bathtub_46_0.urdf', 'kitchen_0-sink_77_0.urdf', 'bathroom_0-toilet_47_0.urdf', 'kitchen_0-bottom_cabinet_66_0.urdf', 'living_room_0-bottom_cabinet_28_0.urdf', 'bedroom_1-chest_13_0.urdf', 'kitchen_0-fridge_61_0.urdf', 'corridor_0-carpet_51_0.urdf', 'kitchen_0-shelf_83_0.urdf', 'kitchen_0-dishwasher_76_0.urdf', 'living_room_0-sofa_chair_24_0.urdf', 'bedroom_0-bottom_cabinet_0_0.urdf', 'bedroom_2-bed_37_2.urdf', 'living_room_0-carpet_53_0.urdf', 'kitchen_0-bottom_cabinet_no_top_70_0.urdf', 'kitchen_0-oven_68_0.urdf'] 

# loading data dict from file path
load_dict_from_file = lambda fp: json.load(open(os.path.join(root_dir, fp.split('|')[0], 
                                                'baseline_phasic_oracle',
                                                'csr',
                                                fp.split('|')[1])))

# load resnet network
resnet_backbone = FeatureLearner(
                    in_channels=5,
                    channel_width=64,
                    pretrained=True,
                    num_classes=0,
                    backbone_str='resnet18')

for fil in tqdm(files_scene):
    path = os.path.join(f'/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/{scene}/baseline_phasic_oracle/csr', fil)
    with open(path) as f:
        data = json.load(f)

    for item_x in data['items']: # each item = one image collected in fil

        obj_key_x = item_x['obj_key']
        # filter
        if obj_key_x not in object_key_filter: continue

        if obj_key_x not in entity_keys:
            entity_keys.append(obj_key_x)

        if obj_key_x not in key_to_frame_dict.keys():
            key_to_frame_dict[obj_key_x] = dict()

        for item_y in data['items']:

            obj_key_y = item_y['obj_key']
            # filter
            if obj_key_y not in object_key_filter: continue

            print('processing: ', obj_key_x, obj_key_y)

            file_path_xy = f'{scene}|{path}' # file path 
            short_path_xy = '{}|{}'.format(file_path_xy.split('|')[0], 
                                           file_path_xy.split('/')[-1])

            # loading data_dict from path
            data_dict_xy = load_dict_from_file(file_path_xy)

            # load image
            image_np = np.array(data_dict_xy['rgb'], dtype=np.uint8)

            # load bounding boxes
            bb_x = [item for item in data_dict_xy['items'] if item['obj_key']==obj_key_x][0]['bounding_box']
            bb_y = [item for item in data_dict_xy['items'] if item['obj_key']==obj_key_y][0]['bounding_box']

            bounding_boxes = dict({'obj_x': bb_x, 'obj_y': bb_y})

            # compute mask and transformed image
            resnet_input = {'mask_1': get_mask_from_bb(bb_x),
                    'mask_2': get_mask_from_bb(bb_y), 
                    'image': Image.fromarray(image_np),
                    'is_self_feature': obj_key_x==obj_key_y,
                    }
            A.TestTransform(resnet_input) # [resize masks + image, normalize image]

            # generate resnet output
            resnet_output = compute_resnet_features(resnet_backbone, resnet_input)

            image_path, resnet_output_path = \
                save_image_resnet(image_np, resnet_output,
                                  obj_key_x, obj_key_y, short_path_xy)

            return_tuple = (file_path_xy, image_path, bounding_boxes, resnet_output_path)

            if obj_key_y not in key_to_frame_dict[obj_key_x]:
                key_to_frame_dict[obj_key_x][obj_key_y] = [return_tuple]

            else:
                key_to_frame_dict[obj_key_x][obj_key_y].append(return_tuple)

arr = np.zeros((len(object_key_filter), len(object_key_filter), len(files_all)))

for x in range(len(object_key_filter)): # x = index, object_key_filter[x] = {x}th entity 
    for y in range(len(object_key_filter)):
        for fil in key_to_frame_dict[object_key_filter[x]][object_key_filter[y]]:

            short_path_xy = '{}|{}'.format(fil.split('|')[0], fil.split('/')[-1])
            arr[x, y, files_all.index(short_path_xy)] = 1

    torch.save({
        'files': files_all,
        'indexing_array': arr,
        'obj_key_to_all_files': key_to_frame_dict,
        'filtered_objects_receptacles': object_key_filter
    }, os.path.join(target_dir, f"{scene}_indices_byobjkey_{timestampStr}_progress{x}.pt"))