import argparse
import os
import torch
import sys
import numpy as np
import torch.nn.functional as F
sys.path.append('/coc/flash5/mpatel377/repos/housekeep/CSR')
sys.path.append('/srv/rail-lab/flash5/mpatel377/repos/housekeep/CSR')
from shared.data_split import DataSplit
from lightning.modules.ranking_module import MLP
from lightning.modules.feature_decoder_module import FeatureDecoderModule
from dataloaders.preference_dataset import PreferenceDataset
from dataloaders.contrastive_dataset import object_key_filter, obj_episode_filters

ROOT = '/coc/'

GROUND_TRUTH_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'
if not os.path.exists(GROUND_TRUTH_FILE): 
    GROUND_TRUTH_FILE.replace('coc','srv/rail-lab')
    GROUND_TRUTH_NAMES_FILE.replace('coc','srv/rail-lab')
    ROOT = '/srv/rail-lab/'


def find_best_ckpt(csr_ckpt_dir):
    best_ckpt = [f for f in os.listdir(csr_ckpt_dir) if f.endswith('.ckpt')]
    best_ckpt.sort(key=lambda x: float(x.split('.')[0].split('-')[-1].split('=')[1]))
    best_ckpt = best_ckpt[0]
    return os.path.join(csr_ckpt_dir, best_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='flash5/kvr6/dev/data/csr_full_v2_test_26-04-2023_13-51-28')
    parser.add_argument('--ckpt_dir_csr', type=str, default='checkpoints/model')
    parser.add_argument('--ckpt_dir_rank', type=str, default='checkpoints/rank_pred_clip/')
    
    args = parser.parse_args()

    ranker = MLP(1024, {'hidden_size':2048})
    print(MLP.__dict__)

    test_data = PreferenceDataset(ROOT+args.data_dir, data_split=DataSplit.TEST, test_unseen_objects=False, csr_ckpt_path=find_best_ckpt(args.ckpt_dir_csr))
    
    obj_names= torch.load(GROUND_TRUTH_NAMES_FILE)['objects']
    assert (obj_names == object_key_filter), "Object names in ground truth file do not match with object names in code."

    try:
        gt_edges = torch.load(GROUND_TRUTH_FILE)
    except:
        gt_edges = torch.load(GROUND_TRUTH_FILE.replace('srv/rail-lab','coc'))

    try:
        files_list = torch.load(GROUND_TRUTH_NAMES_FILE)['files']
    except:
        files_list = torch.load(GROUND_TRUTH_NAMES_FILE.replace('srv/rail-lab','coc'))['files']
    
    files_list = [f.replace('.json','').replace('obs_','') for f in files_list]
    episodes_list = [int(f)//1000 for f in files_list]

    _train_obj = lambda o: (0 <= o < 92) or o > 105
    _val_obj = lambda o: (90 <= o < 95) or o > 105
    average_accuracy = {'seen':[],'unseen':[],'total':[]}
    total_correct = {'seen':0,'unseen':0,'total':0}
    total_total = {'seen':0,'unseen':0,'total':0}
    
    while True:
        data = test_data.get_next_episode()
        if data is None: break
        ep_corr = {'seen':0,'unseen':0,'total':0}
        ep_total = {'seen':0,'unseen':0,'total':0}
        for obj in data:
            pred = None
            pred_rank = -float('inf')
            gt = []
            for rec in data[obj]:
                rank = ranker(data[obj][rec][0])
                label = data[obj][rec][1]
                if rank > pred_rank:
                    pred = rec
                    pred_rank = rank
                if label == 1:
                    gt.append(rec)
            corr = 1 if pred in gt else 0
            if _train_obj(obj):
                total_total['seen'] += 1
                total_correct['seen'] += corr
                ep_total['seen'] += 1
                ep_corr['seen'] += corr
            else:
                total_total['unseen'] += 1
                total_correct['unseen'] += corr
                ep_total['unseen'] += 1
                ep_corr['unseen'] += corr
            total_total['total'] += 1
            total_correct['total'] += corr
            ep_total['total'] += 1
            ep_corr['total'] += corr
        average_accuracy['seen'].append(float(ep_corr['seen']/ep_total['seen']) if ep_total['seen'] > 0 else 0)
        average_accuracy['unseen'].append(float(ep_corr['unseen']/ep_total['unseen']) if ep_total['unseen'] > 0 else 0)
        average_accuracy['total'].append(float(ep_corr['total']/ep_total['total']) if ep_total['total'] > 0 else 0)

        
    print("\n\n")
    print("~~~~~~~~~~~~~~~~~~~~~FINAL RESULTS~~~~~~~~~~~~~~~~~~~~~")
    print("AVERGE ACCURACY (MACRO AVG.) : ",sum(average_accuracy['total'])/len(average_accuracy['total']))
    print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['total']/total_total['total']), f"({total_correct['total']}/{total_total['total']})")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
    print("\n\n")
    print("~~~~~~~~~~~~~~~~~~~~~SEEN RESULTS~~~~~~~~~~~~~~~~~~~~~")
    print("AVERGE ACCURACY (MACRO AVG.) : ",sum(average_accuracy['seen'])/len(average_accuracy['seen']))
    print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['seen']/total_total['seen']), f"({total_correct['seen']}/{total_total['seen']})")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
    print("\n\n")
    print("~~~~~~~~~~~~~~~~~~~~~UNSEEN RESULTS~~~~~~~~~~~~~~~~~~~~~")
    print("AVERGE ACCURACY (MACRO AVG.) : ",sum(average_accuracy['unseen'])/len(average_accuracy['unseen']))
    print("AVERGE ACCURACY (MICRO AVG.) : ",float(total_correct['unseen']/total_total['unseen']), f"({total_correct['unseen']}/{total_total['unseen']})")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n\n")
