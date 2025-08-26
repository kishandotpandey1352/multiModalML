#!/usr/bin/env bash
#SBATCH --job-name=tabqa-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --chdir=/users/acp24kp/finetuning/multiModalML
#SBATCH --output=/users/acp24kp/finetuning/multiModalML/logs/tabqa-%j.out
#SBATCH --error=/users/acp24kp/finetuning/multiModalML/logs/tabqa-%j.err

set -euo pipefail
source /users/acp24kp/anaconda3/etc/profile.d/conda.sh
conda activate mml_env
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

# (Optional) pass absolute paths in config via env vars
python -u fine_tune_tab_qa.py
