#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:0
#SBATCH --job-name="hw7 test"
#SBATCH --output=slurm-%j.out
#SBATCH --mem=32G

module load anaconda
conda activate myenv # activate the Python environment

# runs your code
python -u ../src/gpt2_finetune_but_classification_backbone.py
