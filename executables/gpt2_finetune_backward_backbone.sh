#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu
#SBATCH --job-name="test_gpus"
#SBATCH --output="gpt2_finetune_backbone_alpaca_backward.txt" # Path to store logs

module load anaconda
conda activate myenv # activate the Python environment

# runs your code
python -u ../src/gpt2_finetune_but_classification_backbone.py --is_backward
