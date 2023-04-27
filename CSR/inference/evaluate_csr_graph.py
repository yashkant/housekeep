import argparse
import os
import torch
import sys
import numpy as np
import torch.nn.functional as F
sys.path.append('/coc/flash5/mpatel377/repos/housekeep/CSR')
sys.path.append('/srv/rail-lab/flash5/mpatel377/repos/housekeep/CSR')
from shared.data_split import DataSplit
from lightning.modules.moco2_module_mini import MocoV2Lite
from lightning.modules.feature_decoder_module import FeatureDecoderModule
from dataloaders.contrastive_dataset import ContrastiveDataset
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
    parser.add_argument('--ckpt_dir_edge', type=str, default='checkpoints/edge_pred/')
    
    args = parser.parse_args()
    
    CSRprocess = MocoV2Lite().load_from_checkpoint(find_best_ckpt(args.ckpt_dir_csr))
    CSRprocess.eval()
    get_csr = lambda resnet_vec: torch.mean(torch.stack([torch.nn.functional.normalize(CSRprocess.projection_q(vec), dim=-1) for vec in resnet_vec], dim=0), dim=0)

    edge_process = FeatureDecoderModule().load_from_checkpoint(find_best_ckpt(args.ckpt_dir_edge))
    get_edge = lambda csr_vec: F.softmax(edge_process(csr_vec), dim=-1)[1]

    test_data = ContrastiveDataset(ROOT+args.data_dir, None, DataSplit.TEST, test_unseen_objects=False)
    
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

    average_accuracy = {'seen':[],'unseen':[],'total':[]}
    total_correct = {'seen':0,'unseen':0,'total':0}
    total_total = {'seen':0,'unseen':0,'total':0}
    
    while True:
        data, episode = test_data.get_next_episode()
        if data is None: break
        pred_edges_episode = torch.zeros((len(object_key_filter),len(object_key_filter)))
        seen_entities = torch.zeros(len(object_key_filter))
        seen_map = torch.zeros((len(object_key_filter), len(object_key_filter)))
        for o1 in range(106):
            for o2 in range(106,len(object_key_filter)):
                if len(data[o1][o2]) > 0:
                    seen_map[o1,o2] = 1
                    seen_entities[o1] = 1
                    seen_entities[o2] = 1
                    edge_predicted = get_edge(get_csr(data[o1][o2]).mean(0))
                    pred_edges_episode[o1,o2] = edge_predicted
        gt_edges_episode = gt_edges[episodes_list.index(episode),:,:]
        assert (gt_edges_episode[106:,:]==0).all(), "Ground truth file has edges from receptacles."
        assert (gt_edges_episode[:106,:106]==0).all(), "Ground truth file has object to object edges."

        pred_edges_episode = pred_edges_episode[:106,106:]
        gt_edges_episode = gt_edges_episode[:106,106:]
        seen_map = seen_map[:106,106:]
        present_objects = gt_edges_episode.sum(-1)>0

        seen_receptacles = seen_entities[106:]
        seen_objects = seen_entities[:106]

        seen_and_present_objects = torch.bitwise_and(seen_objects.to(bool), present_objects)
        seen_and_present_object_count = (seen_and_present_objects).sum()

        assert not torch.bitwise_and(seen_objects.to(bool), torch.bitwise_not(present_objects)).any(), f"Seen objects are not present in ground truth. {np.argwhere(seen_objects)} v.s. {np.argwhere(present_objects)}"

        if seen_and_present_object_count < present_objects.sum():
            print(f"Ignoring {present_objects.sum() - seen_and_present_object_count} objects not seen by the agent", end = '.....')

        seen_objects_or_receptacles = seen_objects.unsqueeze(-1).repeat(1,seen_receptacles.size()[0]) * seen_receptacles.unsqueeze(0).repeat(seen_objects.size()[0],1)
        unseen_objects_or_receptacles = (1-seen_objects_or_receptacles).bool()

        pred_edges_episode = F.one_hot(pred_edges_episode.argmax(-1), num_classes=pred_edges_episode.size()[-1]).float()
        correct = (pred_edges_episode[seen_and_present_objects].argmax(-1) == gt_edges_episode[seen_and_present_objects].argmax(-1)).sum()
        total = seen_and_present_objects.sum()


        total_correct['total'] += correct
        total_total['total'] += total

        accuracy = float(correct/total) if seen_and_present_object_count > 0 else 0
        print(f"{accuracy} = {correct}/{total}")
        average_accuracy['total'].append(accuracy)

        accuracy = float(correct/total) if seen_and_present_object_count > 0 else 0
        print(f"{accuracy} = {correct}/{total}")
        average_accuracy['total'].append(accuracy)

        _train_obj, _ = obj_episode_filters(data_split=DataSplit.TRAIN)
        _val_obj, _ = obj_episode_filters(data_split=DataSplit.VAL)
        # _seen_obj_mask = torch.tensor([1 if (_train_obj(o) or _val_obj(o)) else 0 for o in range(106)])[seen_and_present_objects]
        # _unseen_obj_mask = torch.tensor([1 if not (_train_obj(o) or _val_obj(o)) else 0 for o in range(106)])[seen_and_present_objects]
        _seen_obj_mask = torch.tensor([1 if (_train_obj(o)) else 0 for o in range(106)])[seen_and_present_objects]
        _unseen_obj_mask = torch.tensor([1 if not (_train_obj(o)) else 0 for o in range(106)])[seen_and_present_objects]
        
        correct_seen = ((pred_edges_episode[seen_and_present_objects].argmax(-1) == gt_edges_episode[seen_and_present_objects].argmax(-1)) * _seen_obj_mask).sum()
        total_seen = _seen_obj_mask.sum()
        accuracy = float(correct_seen/total_seen) if total_seen > 0 else 0
        total_correct['seen'] += correct_seen
        total_total['seen'] += total_seen
        average_accuracy['seen'].append(accuracy)

        correct_unseen = ((pred_edges_episode[seen_and_present_objects].argmax(-1) == gt_edges_episode[seen_and_present_objects].argmax(-1)) * _unseen_obj_mask).sum()
        total_unseen = _unseen_obj_mask.sum()
        accuracy = float(correct_unseen/total_unseen) if total_unseen > 0 else 0
        total_correct['unseen'] += correct_unseen
        total_total['unseen'] += total_unseen
        average_accuracy['unseen'].append(accuracy)


        
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
