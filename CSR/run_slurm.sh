#!/bin/bash
#SBATCH --job-name=csr_5_obj
#SBATCH --output=slurm_logs/csr_5_obj-%j.out
#SBATCH --error=slurm_logs/csr_5_obj-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --signal=USR1@1000
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --exclude=ig-88,perseverance

source /srv/rail-lab/flash5/kvr6/csrremote.sh
conda activate csr

cd /srv/rail-lab/flash5/kvr6/dev/housekeep_csr/CSR

config="/srv/rail-lab/flash5/kvr6/dev/housekeep_csr/CSR/configs/scenes_skynet_kartik.yml"

echo "In CSR"
python train_csr.py --conf $config