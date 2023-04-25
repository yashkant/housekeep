import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from shared.utils import get_box
from lightning.modules.moco2_module import MocoV2


class RunCSR():
    def __init__(self, 
                root_dir,
                output_dir,
                ckpt_dir,
                feature_size: int = 512,
                csr_datadir = '/srv/flash1/gchhablani3/housekeep/csr_raw/beechwood_0_int/baseline_phasic_oracle/csr'):
        self.root_dir = root_dir
        self.output_dir = output_dir
        scenes_indices = torch.load(os.path.join(root_dir, 'all_scenes_indices.pt'))
        self.iids = scenes_indices['iids']
        self.scenes_indices_arr = scenes_indices['arr']
        self._csr_datadir = csr_datadir

        print("Loading CSR model from :"+ckpt_dir)
        self.CSRencoder = MocoV2.load_from_checkpoint(ckpt_dir).encoder_q
        self.CSGgraph = torch.zeros((len(self.iids), len(self.iids), feature_size))

    def run_obj_pair(self, files, iid1, iid2):
        latents = []
        for file_path in files:
            with open(os.path.join(self._csr_datadir, file_path.split('|')[-1])) as f:
                file_dict = json.load(f)
            item_obj_1 = [item for item in file_dict['items'] if item['iid']==iid1]
            item_obj_2 = [item for item in file_dict['items'] if item['iid']==iid2]
            
            xmin, ymin, xmax, ymax = item_obj_1[0]['bounding_box']
            box_1 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m1 = get_box(box_1)

            xmin, ymin, xmax, ymax = item_obj_2[0]['bounding_box']
            box_2 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m2 = get_box(box_2)

            image = Image.fromarray(np.array(file_dict['rgb'], dtype=np.uint8))

            img_q = torch.cat((image, m1, m2), 1)
            q = self.encoder_q(img_q)  # queries: NxC
            q = torch.nn.functional.normalize(q, dim=1)
            latents.append(q)
        return torch.mean(torch.stack(latents), dim=0)


    def run_obj_pair_resnet_only(self, files, iid1, iid2):
        latents = []
        for file_path in files:
            with open(os.path.join(self._csr_datadir, file_path.split('|')[-1])) as f:
                file_dict = json.load(f)
            item_obj_1 = [item for item in file_dict['items'] if item['iid']==iid1]
            item_obj_2 = [item for item in file_dict['items'] if item['iid']==iid2]
            
            xmin, ymin, xmax, ymax = item_obj_1[0]['bounding_box']
            box_1 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m1 = get_box(box_1)

            xmin, ymin, xmax, ymax = item_obj_2[0]['bounding_box']
            box_2 = np.array([[max(xmin-5, 0), max(ymin-5, 0)], [min(xmax+5+1, 255), min(ymax+5+1, 255)]])
            m2 = get_box(box_2)

            image = Image.fromarray(np.array(file_dict['rgb'], dtype=np.uint8))

            img_q = torch.cat((image, m1, m2), 1)
            q = self.encoder_q(img_q)  # queries: NxC
            q = torch.nn.functional.normalize(q, dim=1)
            latents.append(q)
        return torch.mean(torch.stack(latents), dim=0)


    def run(self):
        for idx1, iid1 in enumerate(self.iids):
            for idx2, iid2 in enumerate(self.iids):
                files = [self.scenes_indices['files'][fileidx] for fileidx in self.scenes_indices_arr]
                self.CSGgraph[idx1, idx2] = self.run_obj_pair(files, iid1, iid2)
        torch.save(self.CSGgraph, os.path.join(self.output_dir, 'CSGgraph.pt'))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    args = parser.parse_args()
    run_csr = RunCSR(args.root_dir, args.output_dir, args.ckpt_dir)
    run_csr.run()
    