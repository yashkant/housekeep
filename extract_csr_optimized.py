import os
import json
import numpy as np
import torchvision
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set the root directory where you want to start the script
root_dir = './csr_raw'

# include_scene_names = ['benevolence_2_int', 'wainscott_0_int', 'beechwood_0_int', 'rs_int']
include_scene_names = 'beechwood_1_int  benevolence_1_int  ihlen_0_int  ihlen_1_int  merom_0_int  merom_1_int  pomaria_0_int  pomaria_1_int  pomaria_2_int  wainscott_1_int'.replace('  ', ' ').split(' ')
# Walk through all subdirectories in the root directory
for subdir, dirs, files in os.walk(root_dir):
    if len(subdir.split('/')) >=3 and subdir.split('/')[-3] not in include_scene_names:
        print("Skipping ", subdir.split('/')[-3])
        continue
    # Check if the subdirectory has any obs_{idx}.json files
    obs_files = [f for f in files if f.startswith('obs_') and f.endswith('.json')]
    if len(obs_files) > 0:
        print("Working on ", subdir.split('/')[-3])
        # Create the 'csr' directory in the second parent of the subdirectory
        parent_dir = os.path.dirname(subdir)
        csr_dir = os.path.join(parent_dir, 'csr')
        os.makedirs(csr_dir, exist_ok=True)
        # Load the obs_{idx}.json files one by one
        for obs_file in tqdm(obs_files):  
            obs_file_path = os.path.join(subdir, obs_file)
            
            csr_file = f"csr_{obs_file.split('_')[1]}"
            csr_file_path = os.path.join(csr_dir, csr_file)
            
            csr_data = None
            if os.path.exists(csr_file_path):
                try:            
                    with open(csr_file_path) as f:
                        csr_data = json.load(f)
                except:
                    csr_data = None
                
            if csr_data and len(csr_data['items']) !=0 and 'current_mapping' in csr_data:
                print("Skipping this observation")
                continue
            
            # print(obs_file_path)
            with open(obs_file_path, 'r') as f:
                obs_data = json.loads(json.load(f))
            # Extract the specific values you want from the loaded json
            # Create a new json with the extracted values
            if csr_data and len(csr_data['items']) !=0:
                print("Just adding mappings")
                csr_data['current_mapping'] = obs_data[0]['cos_eor']['current_mapping']
                csr_data['correct_mapping'] = obs_data[0]['cos_eor']['correct_mapping']
            else:
                print("Adding items and mappings")
                csr_data = {'rgb': obs_data[0]['rgb'], 'mask': obs_data[0]['semantic'], 'depth': obs_data[0]['depth'], 'items': [], 'current_mapping': obs_data[0]['cos_eor']['current_mapping'], 'correct_mapping': obs_data[0]['cos_eor']['correct_mapping']}
                segmentation_masks = []
                for iid in np.unique(obs_data[0]['semantic']):
                    # Semantic Map has IID, IID to SIM OBJ ID gives the correct class
                    if iid != 0:
                        item = {'iid': int(iid)}
                        sim_id = obs_data[0]['cos_eor']['iid_to_sim_obj_id'][str(iid)]
                        item['sim_id'] = sim_id
                        segmentation_mask = np.array(obs_data[0]['semantic'])==iid
                        segmentation_masks.append(segmentation_mask)
                        # segmented_image = np.where(segmentation_mask, np.array(obs_data[0]['rgb']), np.zeros_like(obs_data[0]['rgb']))
                        
                        # # Plot Segmented Image
                        # plt.imshow(segmented_image)
                        # plt.show()

                        # print(iid, sim_id)
                        item['obj_key'] = obs_data[0]['cos_eor']['sim_obj_id_to_obj_key'][str(sim_id)]
                        item['type'] = obs_data[0]['cos_eor']['sim_obj_id_to_type'][str(sim_id)]

                        csr_data['items'].append(item)

                boxes = torchvision.ops.masks_to_boxes(torch.tensor(np.array(segmentation_masks)).squeeze())         
                # bounding_box = draw_bounding_boxes(torch.tensor(obs_data[0]['rgb'], dtype=torch.uint8).permute(2, 0, 1), boxes, colors="red")
                # show(bounding_box)
                # plt.show()
                
                for item, box in zip(csr_data['items'], boxes):
                    # print(item)
                    # print(box)
                    item['bounding_box'] = tuple(map(int, box))
                    # Plot Cropped Image
                    xmin, ymin, xmax, ymax = item['bounding_box']
                    cropped_image = np.array(obs_data[0]['rgb'])[ymin:ymax+1, xmin:xmax+1]
                    # cropped_image = np.array(obs_data[0]['rgb'])[max(0, ymin-5):min(ymax+1+5, 512), max(0, xmin-5):min(xmax+1+5, 512)]
                    item['cropped_image'] = cropped_image.tolist()
                    # plt.imsave(f"{item['obj_key']}_{item['iid']}.jpg", cropped_image.astype(np.uint8))
                    # plt.show()
                
                        # Deets
                        # print(boxes)
                with open(csr_file_path, 'w') as f:
                    json.dump(csr_data, f)