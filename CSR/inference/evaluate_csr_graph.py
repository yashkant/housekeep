import argparse
import os
import torch
import sys
import torch.nn.functional as F
sys.path.append('/coc/flash5/mpatel377/repos/housekeep/CSR')
from shared.data_split import DataSplit
from lightning.modules.moco2_module_mini import MocoV2Lite
from lightning.modules.feature_decoder_module import FeatureDecoderModule
from dataloaders.contrastive_dataset import ContrastiveDataset
from dataloaders.contrastive_dataset import object_key_filter

GROUND_TRUTH_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_complete.pt'
GROUND_TRUTH_NAMES_FILE = '/coc/flash5/mpatel377/data/csr/scene_graph_edges_names.pt'

def find_best_ckpt(csr_ckpt_dir):
    best_ckpt = [f for f in os.listdir(csr_ckpt_dir) if f.endswith('.ckpt')]
    best_ckpt.sort(key=lambda x: float(x.split('.')[0].split('-')[-1].split('=')[1]))
    best_ckpt = best_ckpt[0]
    return os.path.join(csr_ckpt_dir, best_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/coc/flash5/kvr6/dev/data/csr_full_v2_25-04-2023_22-06-27')
    parser.add_argument('--ckpt_dir_csr', type=str, default='checkpoints/model/')
    parser.add_argument('--ckpt_dir_edge', type=str, default='checkpoints/edge_pred/')
    parser.add_argument('--output_dir', type=str, default='/coc/flash5/mpatel377/data/csr_eval')
    parser.add_argument('--test_unseen_objects', action='store_true')
    
    args = parser.parse_args()
    
    CSRprocess = MocoV2Lite().load_from_checkpoint(find_best_ckpt(args.ckpt_dir_csr))
    CSRprocess.eval()
    get_csr = lambda resnet_vec: torch.mean(torch.stack([torch.nn.functional.normalize(CSRprocess.projection_q(vec), dim=-1) for vec in resnet_vec], dim=0), dim=0)

    edge_process = FeatureDecoderModule().load_from_checkpoint(find_best_ckpt(args.ckpt_dir_edge))
    get_edge = lambda csr_vec: F.softmax(edge_process(csr_vec), dim=-1)[1]

    test_data = ContrastiveDataset(args.data_dir, None, DataSplit.TRAIN, test_unseen_objects=args.test_unseen_objects)
    
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



    average_accuracy = []
    while True:
        data, episode = test_data.get_next_episode()
        if data is None: break
        pred_edges_episode = torch.zeros((len(object_key_filter),len(object_key_filter)))
        seen_objects = torch.zeros(106)
        for o1 in range(106):
            for o2 in range(106,len(object_key_filter)):
                if len(data[o1][o2]) > 0:
                    # print(o1,o2)
                    seen_objects[o1] = 1
                    edge_predicted = get_edge(get_csr(data[o1][o2]).mean(0))
                    pred_edges_episode[o1,o2] = edge_predicted
        gt_edges_episode = gt_edges[episodes_list.index(episode),:,:]
        pred_edges_episode = pred_edges_episode[:106,106:]
        gt_edges_episode = gt_edges_episode[:106,106:]
        present_objects = gt_edges_episode.sum(-1)>0
        # print(seen_objects)
        # print(seen_objects)
        # print(gt_edges_episode.argmax(-1))
        # print(pred_edges_episode.argmax(-1))
        pred_edges_episode = F.one_hot(pred_edges_episode.argmax(-1), num_classes=pred_edges_episode.size()[-1]).float()
        
        # print(pred_edges_episode.size(), gt_edges_episode.size())
        # print((pred_edges_episode[present_objects].argmax(-1) == gt_edges_episode[present_objects].argmax(-1)).sum())
        # print((pred_edges_episode[present_objects]).argmax(-1))
        # print((gt_edges_episode[present_objects]).argmax(-1))

        accuracy = float((pred_edges_episode[present_objects].argmax(-1) == gt_edges_episode[present_objects].argmax(-1)).sum()/present_objects.sum())
        print(f"{accuracy} = {(pred_edges_episode[present_objects].argmax(-1) == gt_edges_episode[present_objects].argmax(-1)).sum()}/{present_objects.sum()}")
        average_accuracy.append(accuracy)
    print("AVERGE ACCURACY : ",sum(average_accuracy)/len(average_accuracy))
