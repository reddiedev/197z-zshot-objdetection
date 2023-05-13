#!/bin/bash
#SBATCH --output=notMNIST.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=samsung

#Activate your environment, source command is required

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/sean_red_mendoza/cuda-test

# Insert your code here.
nvcc --version
srun python test.py
srun bash -c 'CUDA_VISIBLE_DEVICES=0 python test.py --workers 1'
srun bash -c 'CUDA_VISIBLE_DEVICES=1 python test.py --workers 1'

# srun bash -c 'CUDA_VISIBLE_DEVICES=0 python explore-sam.py  --workers 1 --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction RCNN --SequenceModeling None --Prediction Attn'
# srun jupyter nbconvert --to notebook --execute explore-sam.ipynb

# [Optional] You can run other commands here

# Sample training code format
srun nvidia-smi
srun sleep 10
# srun bash -c 'CUDA_VISIBLE_DEVICES=0 python train.py --workers 1 --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction RCNN --SequenceModeling None --Prediction Attn'