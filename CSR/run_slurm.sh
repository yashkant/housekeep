#!/bin/bash
#SBATCH --job-name=csr_train
#SBATCH --output=slurm_logs/csr_train-%j.out
#SBATCH --error=slurm_logs/csr_train-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --constraint=a40

source /srv/rail-lab/flash5/kvr6/csrremote.sh
conda activate csr

cd /srv/rail-lab/flash5/mpatel377/repos/housekeep/CSR

config="configs/scenes_skynet_kartik.yml"

echo "In CSR"
python train_edge.py --conf $config
