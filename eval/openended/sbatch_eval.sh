#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=8g
#SBATCH -J "vicuna real data"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100

nvidia-smi
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY before running}"
python3 -m fastchat.serve.controller & python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5-16k & python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 &
sleep 180; curl http://localhost:8000/v1/models; echo "Running scripts now!"; python3 /home/simran/SLU/eval/openended/quality_scoring_vicuna_tune.py
python3 /home/simran/SLU/eval/openended/interclass_scorer.py