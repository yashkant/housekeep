#! /bin/bash

#SBATCH -o slurm_output_%j.txt
#SBATCH -e slurm_err_%j.txt
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J copy_data
#SBATCH -p short

source /srv/rail-lab/flash5/kvr6/csrremote.sh
source activate csr

python postpostprocess_index.py
