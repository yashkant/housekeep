import json
import os
import numpy as np
import random
from tqdm import tqdm

from PIL import Image
from shared.data_split import DataSplit
import torch
from torch.utils.data import Dataset
from lightning.modules.moco2_module_mini import MocoV2Lite
from transformers import CLIPImageProcessor
from dataloaders.contrastive_dataset import object_key_filter, obj_episode_filters 


GROUND_TRUTH_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/srv/rail-lab/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'

MAP_FILE = '/coc/flash5/mpatel377/data/csr/preference_map_universal.pt'

def get_preference(item1, item2, episode_map, persona=None):
    if persona is not None:
        raise NotImplementedError("Persona not implemented yet")
    else:
        if item2['obj_key'] in episode_map[item1['obj_key']]:
            label = 1
        else:
            label = 0
    return label

class PreferenceDataset(Dataset):

    def __init__(self, 
                 root_dir: str, 
                 data_split: DataSplit,
                 csr_ckpt_path: str,
                 image_input: bool = False,
                 max_annotations = 100000,
                 test_unseen_objects = True
                 ):
     
       
        use_obj, use_episode = obj_episode_filters(data_split, test_unseen_objects)
        
        self.CSRprocess = MocoV2Lite().load_from_checkpoint(csr_ckpt_path)
        self.CSRprocess.eval()
        get_csr = lambda resnet_vec: torch.nn.functional.normalize(self.CSRprocess.projection_q(resnet_vec), dim=-1)
        self.clip_path = '/srv/rail-lab/flash5/mpatel377/data/csr_clips/'
        
        try:
            gt_edges = torch.load(GROUND_TRUTH_FILE)
        except:
            gt_edges = torch.load(GROUND_TRUTH_FILE.replace('srv/rail-lab','coc'))

        try:
            files_list = torch.load(GROUND_TRUTH_NAMES_FILE)['files']
            obj_list = torch.load(GROUND_TRUTH_NAMES_FILE)['objects']
        except:
            files_list = torch.load(GROUND_TRUTH_NAMES_FILE.replace('srv/rail-lab','coc'))['files']
            obj_list = torch.load(GROUND_TRUTH_NAMES_FILE.replace('srv/rail-lab','coc'))['objects']
            self.clip_path = '/coc/flash5/mpatel377/data/csr_clips/'

        self.final_map = {}
        seen_episodes = []
        
        def make_final_map():
            if os.path.exists(MAP_FILE):
                self.final_map = torch.load(MAP_FILE)
                return
            datadir = '/srv/cvmlp-lab/flash1/gchhablani3/housekeep/csr_raw/ihlen_1_int/baseline_phasic_oracle/observations'
            for file in tqdm(os.listdir(datadir)):
                episode = int(file.replace('.json','').split('_')[-1])//1000
                if episode not in seen_episodes:
                    seen_episodes.append(episode)
                    data = json.loads(json.load(open(os.path.join(datadir, file))))[0]['cos_eor']['correct_mapping']
                    for o,rr in data.items():
                        if o in obj_list:
                            if obj_list.index(o) not in self.final_map: self.final_map[obj_list.index(o)] = []
                            self.final_map[obj_list.index(o)] += [obj_list.index(r) for r in rr if r in obj_list]
            torch.save(self.final_map, MAP_FILE)


        self.resnet_path = os.path.join(root_dir, 'ihlen_1_int', 
                                    'baseline_phasic_oracle','resnet'
                                    )



        self.indexed_data = []

        def is_obj(o):
            return o < 106
        
        self.object_clips = {}
        self.receptacle_clips = {}
        self.object_csrs = {}
        self.receptacle_csrs = {}
        
        for clipfile in tqdm(os.listdir(self.clip_path)):
            epstep, objidx = clipfile.replace('.pt','').split('_')
            objidx = int(objidx)
            if use_episode(int(epstep)//1000) and use_obj(objidx):
                if is_obj(objidx):
                    if objidx not in self.object_clips:
                        self.object_clips[objidx] = {}
                    self.object_clips[objidx][int(epstep)]=torch.load(os.path.join(self.clip_path, clipfile)).detach()
                else:
                    if objidx not in self.receptacle_clips:
                        self.receptacle_clips[objidx] = {}
                    self.receptacle_clips[objidx][int(epstep)]=torch.load(os.path.join(self.clip_path, clipfile)).detach()

        print("Loaded clips")

        for resnetfile in tqdm(os.listdir(self.resnet_path)):
            o1, o2, _, episodestep = resnetfile.split('_')
            episodestep = episodestep.replace('.pt','')
            if o2 == o1:
                objidx = int(o1)
                if use_episode(int(episodestep)//1000) and use_obj(objidx):
                    episodestep = int(episodestep.replace('.pt',''))
                    episode = episodestep//1000
                    csr_vec = get_csr(torch.load(os.path.join(self.resnet_path, resnetfile)))
                    if is_obj(objidx):
                        if objidx not in self.object_csrs:
                            self.object_csrs[objidx] = {}
                        self.object_csrs[objidx][int(episodestep)] = csr_vec.detach()
                    else:
                        if objidx not in self.receptacle_csrs:
                            self.receptacle_csrs[objidx] = {}
                        self.receptacle_csrs[objidx][int(episodestep)] = csr_vec.detach()
        
        print("Loaded resnets")

        make_final_map()

        print("Map generated")

        self.indexed_data = []
        for obj in tqdm(self.object_clips):
            for epstep_obj in self.object_clips[obj]:
                if obj in self.object_csrs and epstep_obj in self.object_csrs[obj]:
                    for receptacle in self.receptacle_clips:
                        for epstep_rec in self.receptacle_clips[receptacle]:
                            if receptacle in self.receptacle_csrs and epstep_rec in self.receptacle_csrs[receptacle]:
                                label = torch.tensor([receptacle in self.final_map[obj]])
                                self.indexed_data.append((obj, epstep_obj, receptacle, epstep_rec, label))

        print('Indexed data created!!')

        # self.feature_size = 2048
        self.feature_size = 1024

        print('Total length of dataset type ', data_split, ': ', len(self.indexed_data))
        self.length = min(len(self.indexed_data), max_annotations) if max_annotations > 0 else len(self.indexed_data)
        print('Using length of dataset type ', data_split, ': ', self.length)

        self.is_train = True
        self.episode_curr = -1

    def get_next_episode(self):
        self.episode_curr += 1
        if self.episode_curr >= 10:
            return None
        dataitems = {}
        for datapoint in self.indexed_data:
            obj, epstep_obj, receptacle, epstep_rec, label = datapoint
            if epstep_obj//1000 == self.episode_curr:
                clip_obj = self.object_clips[obj][epstep_obj]
                csr_obj = self.object_csrs[obj][epstep_obj]
                clip_receptacle = self.receptacle_clips[receptacle][epstep_rec]
                csr_receptacle = self.receptacle_csrs[receptacle][epstep_rec]
                dataitem = {'csr_item_1':csr_obj, 'clip_item_1':clip_obj, 'csr_item_2':csr_receptacle, 'clip_item_2':clip_receptacle, 'label':label}
                if obj not in dataitems: dataitems[obj] = {}
                if receptacle not in dataitems[obj]: dataitems[obj][receptacle] = []
                dataitems[obj][receptacle].append(self.collate_fn([dataitem]))
        
        for obj in dataitems:
            for rec in dataitems[obj]:
                dataitems[obj][rec] =  torch.mean(torch.stack([d[0] for d in dataitems[obj][rec]], dim=0), dim=0), dataitems[obj][rec][0][1]
        
        return dataitems

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns : Tuple (data,label)
                    data : (csr_feature1, clip_feature1, csr_feature2, clip_feature2)
                    label: 0 or 1
        """
        obj, epstep_obj, receptacle, epstep_rec, label = self.indexed_data[idx]
        clip_obj = self.object_clips[obj][epstep_obj]
        csr_obj = self.object_csrs[obj][epstep_obj]
        clip_receptacle = self.receptacle_clips[receptacle][epstep_rec]
        csr_receptacle = self.receptacle_csrs[receptacle][epstep_rec]
        dataitem = {'csr_item_1':csr_obj, 'clip_item_1':clip_obj, 'csr_item_2':csr_receptacle, 'clip_item_2':clip_receptacle, 'label':label}
        return dataitem

    def collate_fn(self, data_points):
        ''' Collate function for dataloader.
            Args:
                data_points: A list of dicts'''

        # data_points is a list of dicts
        csr_embbs_1 = torch.cat([torch.tensor(d['csr_item_1']) for d in data_points], dim=0)
        clip_embbs_1 = torch.cat([torch.tensor(d['clip_item_1']) for d in data_points], dim=0)
        csr_embbs_2 = torch.cat([torch.tensor(d['csr_item_2']) for d in data_points], dim=0)
        clip_embbs_2 = torch.cat([torch.tensor(d['clip_item_2']) for d in data_points], dim=0)

        # input_tensor = torch.cat([csr_embbs_1, clip_embbs_1, csr_embbs_2, clip_embbs_2], dim=1)
        input_tensor = torch.cat([clip_embbs_1, clip_embbs_2], dim=1)

        labels = torch.tensor([d['label'] for d in data_points]).float()

        # if self.is_train:
        #     return input_tensor, labels, data_points

        # else:
        return input_tensor, labels
