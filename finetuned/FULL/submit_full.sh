#!/bin/bash
#SBATCH --job-name=gemma_FSDP_2GPU
#SBATCH --output=fsdp_log_%j.out
#SBATCH --error=fsdp_error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2             # <--- CHANGED TO 2 (Fits on 1 node)
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Activate Environment
source /home/shashwatp.singh.cse24.itbhu/gemma_project/venv/bin/activate

# 2. Run Distributed Training (2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 train_full.py