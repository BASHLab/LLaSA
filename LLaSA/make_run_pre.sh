#!/bin/bash
lr=$1
epoch=$2
cat <<EoF
#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=8g
#SBATCH -J "llasa pretrain lr $lr epoch $epoch"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH -C A100

echo "pretrain lr $lr epoch $epoch"
nvidia-smi
bash scripts/v1_5/pretrain_llasa_variable.sh $lr $epoch
EoF
