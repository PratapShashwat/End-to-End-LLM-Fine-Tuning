#!/bin/bash
#SBATCH --job-name=gemma_eval
#SBATCH --output=gemma_eval_%j.out
#SBATCH --error=gemma_eval_%j.err
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

# 3. Load Safe CUDA
module load cuda/11.1

# 4. Activate Environment
source venv/bin/activate

# 5. Define file paths (Make sure INPUT_FILE matches your actual json file name)
INPUT_FILE="./evaluate_summaries/trim_eval.out" 
OUTPUT_FILE="Q_evaluation_results.json"
MODEL_PATH="google/gemma-2-2b-it"

echo "Starting Gemma evaluation job..."

# 6. Run the Evaluation Script
python evaluate_qwen.py \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE \
    --model_path $MODEL_PATH

echo "Job finished."