#!/bin/bash
#SBATCH --job-name=csr_5_obj
#SBATCH --output=slurm_logs/csr_5_obj-%j.out
#SBATCH --error=slurm_logs/csr_5_obj-%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 4
#SBATCH --signal=USR1@1000
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --exclude=ig-88,perseverance

source /srv/flash1/gchhablani3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate csr

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

cd /srv/flash1/gchhablani3/housekeep/CSR

# dataset=$1

# config="configs/experiments/ddppo_instance_imagenav.yaml"

# DATA_PATH="data/datasets/instance_image_nav"
# TENSORBOARD_DIR="tb/iinav/ddppo/long_seed_1"
# CHECKPOINT_DIR="data/new_checkpoints/iinav/ddppo/long_seed_1"

# mkdir -p $TENSORBOARD_DIR
# mkdir -p $CHECKPOINT_DIR
set -x

echo "In CSR"
srun python train_csr.py --conf configs/all_scenes.yml