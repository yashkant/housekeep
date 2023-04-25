import argparse
import os
import torch
from generateCSR import RunCSR
from lightning.modules.feature_decoder_module import FeatureDecoderModule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--ground_truth_file', type=str)
    args = parser.parse_args()
    run_csr = RunCSR(args.root_dir, args.output_dir, args.ckpt_dir)
    run_csr.run() # Saves the CSR graph tensor
    
    CSGgraph = torch.load(os.path.join(args.output_dir, 'CSGgraph.pt'))
    
    fdm = FeatureDecoderModule.load_from_checkpoint(os.path.join(args.ckpt_dir, 'model.ckpt'))
    
    preds = fdm(CSGgraph) # (n, n, 1)
    
    # Calculate accuracy between ground truth and preds
    ground_truth = torch.load(args.ground_truth_file)
    accuracy = ((preds > 0.5).float() == ground_truth).float().mean()
    print(f'Accuracy: {accuracy.item()}')
