import os
import json
import torch
# from transformers import CLIPImageProcessor
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel


target_dir='/srv/rail-lab/flash5/mpatel377/data/csr_clips'
datadir = '/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/ihlen_1_int/baseline_phasic_oracle/csr'

objs = ['dutch_oven_1', 'chocolate_milk_pods_1', 'chocolate_1', 'antidepressant_1', 'wipe_warmer_1', 'sponge_1', 'herring_fillets_1', 'dumbbell_rack_1', 'thermal_laminator_1', 'camera_1', 'soap_dish_1', 'teapot_1', 'bleach_cleanser_1', 'bath_sheet_1', 'gloves_1', 'towel_1', 'string_lights_1', 'skillet_lid_1', 'knife_block_1', 'spoon_rest_1', 'lamp_1', 'fork_1', 'mini_soccer_ball_1', 'salt_shaker_1', 'sushi_mat_1', 'helmet_1', 'condiment_1', 'mustard_bottle_1', 'set-top_box_1', 'weight_loss_guide_1', 'cracker_box_1', 'skillet_1', 'toothbrush_pack_1', 'medicine_1', 'lime_squeezer_1', 'tomato_soup_can_1', 'water_bottle_1', 'candle_holder_1', 'cereal_1', 'diaper_pack_1', 'flashlight_1', 'baseball_1', 'clock_1', 'toaster_1', 'heavy_master_chef_can_1', 'spoon_1', 'handbag_1', 'ramekin_1', 'golf_ball_1', 'umbrella_1', 'plant_saucer_1', 'vase_1', 'fruit_snack_1', 'cake_mix_1', 'dishtowel_1', 'hair_dryer_1', 'dustpan_and_brush_1', 'peppermint_1', 'tea_pods_1', 'saute_pan_1', 'sparkling_water_1', 'chopping_board_1', 'coffee_pods_1', 'candy_bar_1', 'softball_1', 'can_opener_1', 'hat_1', 'lantern_1', 'utensil_holder_1', 'master_chef_can_1', 'portable_speaker_1', 'tablet_1', 'electric_heater_1', 'table_lamp_1', 'candy_1', 'coffee_beans_1', 'light_bulb_1', 'dietary_supplement_1', 'cloth_1', 'sanitary_pads_1', 'spatula_1', 'pitcher_base_1', 'washcloth_1', 'plate_1', 'potted_meat_can_1', 'saucer_1', 'blender_jar_1', 'dish_drainer_1', 'gelatin_box_1', 'soap_dispenser_1', 'fondant_1', 'racquetball_1', 'dumbbell_1', 'coffeemaker_1', 'incontinence_pads_1', 'pan_1', 'laptop_1', 'router_1', 'tennis_ball_1', 'plant_1', 'chocolate_box_1', 'shredder_1', 'electric_toothbrush_1', 'dispensing_closure_1', 'sponge_dish_1', 'tampons_1',
        'bathroom_0-sink_49_0.urdf', 'kitchen_0-counter_80_0.urdf', 'dining_room_0-table_19_0.urdf', 'bedroom_1-chest_12_0.urdf', 'kitchen_0-counter_79_0.urdf', 'living_room_0-coffee_table_26_0.urdf', 'bedroom_0-bed_33_2.urdf', 'bedroom_1-carpet_41_0.urdf', 'dining_room_0-chair_20_0.urdf', 'kitchen_0-top_cabinet_63_0.urdf', 'kitchen_0-counter_65_0.urdf', 'kitchen_0-top_cabinet_62_0.urdf', 'kitchen_0-counter_64_0.urdf', 'dining_room_0-carpet_52_0.urdf', 'bedroom_2-chair_11_0.urdf', 'bedroom_0-bottom_cabinet_1_0.urdf', 'bedroom_1-table_14_0.urdf', 'bathroom_0-bathtub_46_0.urdf', 'kitchen_0-sink_77_0.urdf', 'bathroom_0-toilet_47_0.urdf', 'kitchen_0-bottom_cabinet_66_0.urdf', 'living_room_0-bottom_cabinet_28_0.urdf', 'bedroom_1-chest_13_0.urdf', 'kitchen_0-fridge_61_0.urdf', 'corridor_0-carpet_51_0.urdf', 'kitchen_0-shelf_83_0.urdf', 'kitchen_0-dishwasher_76_0.urdf', 'living_room_0-sofa_chair_24_0.urdf', 'bedroom_0-bottom_cabinet_0_0.urdf', 'bedroom_2-bed_37_2.urdf', 'living_room_0-carpet_53_0.urdf', 'kitchen_0-bottom_cabinet_no_top_70_0.urdf', 'kitchen_0-oven_68_0.urdf'] 

files = os.listdir(datadir)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


for i,file in enumerate(files):
    epstep = file.replace('.json','').split('_')[-1]
    data = json.load(open(os.path.join(datadir, file)))['items']
    for item in data:
        if item['obj_key'] not in objs: 
            print(f"Skipping {item['obj_key']}")
            continue
        objidx = objs.index(item['obj_key'])
        cropped_image = Image.fromarray(np.array(item['cropped_image'], dtype=np.uint8))
        try:
            processed = processor(images=cropped_image, return_tensors="pt", padding=True)
            clip_feature = model.get_image_features(pixel_values=processed['pixel_values'])
            torch.save(clip_feature, f"{target_dir}/{epstep}_{objidx}.pt")
        except ValueError as e:
            print(f"Skipping {item['obj_key']} because {e}")
            continue
