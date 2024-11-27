#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=8g
#SBATCH -J "llasa lr 3e-5 mm_lr 1e-4 epoch 1 lora_r 8 lora_alpha 8 /home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa2-pretrain-lr-1e-3-epoch-1 1"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C V100|A100

nvidia-smi
export OPENAI_API_KEY=
python3 /home/simran/SLU/eval/openended/quality_scoring.py
# python3 /home/simran/SLU/eval/llasa_classification_4_limubert.py
# python3 /home/simran/SLU/eval/llasa_classification_4_limubert_shl.py
# python3 /home/simran/SLU/eval/llasa_classification_4_limubert_mobiact.py
# python3 /home/simran/SLU/eval/llasa_classification_4_limubert_hapt.py
# python3 /home/simran/SLU/eval/human/classify.py
# python3 /home/simran/SLU/eval/gpt35_classification_4_limubert.py
# python3 /home/simran/SLU/eval/gpt35_classification_4_shl_fine_tuned.py
