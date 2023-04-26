import json
import os
import numpy as np

from PIL import Image
from shared.data_split import DataSplit
import torch
from torch.utils.data import Dataset
from lightning.modules.moco2_module import MocoV2
from lightning.modules.moco2_module_mini import MocoV2Lite
from dataloaders.contrastive_dataset import object_key_filter


GROUND_TRUTH_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'


class ReceptacleDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                #  image_input: bool = False,
                 max_annotations = -1,
                 test_unseen_objects = True
                 ):
        
        self.root_dir = root_dir
        self.data_split = data_split

        if data_split == DataSplit.TRAIN:
            use_obj = lambda o: o in np.arange(0,90) or o > 105
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.VAL:
            use_obj = lambda o: o in np.arange(85,95) or o > 105
            use_episode = lambda e: e > 10
        elif data_split == DataSplit.TEST:
            if test_unseen_objects:
                use_obj = lambda o: o in np.arange(95,106) or o > 105
            else:
                use_obj = lambda o: o in np.arange(0,95) or o > 105
            use_episode = lambda e: e <= 10
        else:
            assert False, 'Data split not recognized'
        
        # ## USE ALL OBJECTS AND EPISODES
        # use_obj = lambda o: True
        # use_episode = lambda o: True

        self.CSRprocess = MocoV2Lite().load_from_checkpoint(csr_ckpt_path)
        self.CSRprocess.eval()
        get_csr = lambda resnet_vec: torch.nn.functional.normalize(self.CSRprocess.projection_q(resnet_vec), dim=-1).squeeze(0).detach()
        
        gt_edges = torch.load(GROUND_TRUTH_FILE)
        files_list = torch.load(GROUND_TRUTH_NAMES_FILE)['files']
        files_list = [f.replace('.json','').replace('obs_','') for f in files_list]

        self.resnet_path = os.path.join(root_dir, 'ihlen_1_int', 
                                    'baseline_phasic_oracle','resnet'
                                    )

        self.indexed_data = []

        def is_obj(o):
            return o < 106

        for file in os.listdir(self.resnet_path):
            o1, o2, _, episodestep = file.split('_')
            o1 = int(o1)
            o2 = int(o2)
            episodestep = int(episodestep.replace('.pt',''))
            episode = episodestep//1000
            if use_episode(episode):
                if use_obj(o1) and use_obj(o2):
                    if is_obj(o1) and not is_obj(o2):
                        epstep = file.replace('.pt','').split('_')[-1]
                        fidx = files_list.index(epstep)
                        self.indexed_data.append((get_csr(torch.load(os.path.join(self.resnet_path, file))), gt_edges[fidx, o1, o2].to(int)))

        print('Indexed data created!!')

        print('Total length of dataset type ', data_split, ': ', len(self.indexed_data))
        self.length = max_annotations if max_annotations > 0 else len(self.indexed_data)
        print('Using length of dataset type ', data_split, ': ', self.length)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.indexed_data[idx]
