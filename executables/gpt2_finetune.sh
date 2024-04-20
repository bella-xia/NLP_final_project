#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="test_gpus"
#SBATCH --output="load_model.txt" # Path to store logs

module load anaconda
conda activate myenv # activate the Python environment

# initialize the policy model
python ../src/gpt2_finetune.py