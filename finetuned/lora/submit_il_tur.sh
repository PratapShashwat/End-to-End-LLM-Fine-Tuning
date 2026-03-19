#!/bin/bash
#SBATCH --job-name=iltur_mam
#SBATCH --output=iltur_log_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=3-00:00:00

# 1. Clear modules
module purge

# 2. Load Python
module load python3.8/3.8

# 3. Load Safe CUDA (with a backup plan)
# We try 11.1 (Standard). The "|| true" prevents the job from crashing if it fails.
module load cuda/11.1

# 4. Activate Environment
source venv/bin/activate

# 5. Run the Fail-Safe Python Script
python run_test_only.py
