#!/bin/bash
lr=$1
mm_lr=$2
epoch=$3
lora_r=$4
lora_alpha=$5
adapter_path=$6
adapter_id=$7
cat <<EoF
#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=8g
#SBATCH -J "llasa lr $lr mm_lr $mm_lr epoch $epoch lora_r $lora_r lora_alpha $lora_alpha $adapter_path $adapter_id"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH -C A100

echo "lr $lr mm_lr $mm_lr epoch $epoch lora_r $lora_r lora_alpha $lora_alpha adapter $adapter_path $adapter_id"
nvidia-smi
bash scripts/v1_5/llasa_lora_variable.sh $lr $mm_lr $epoch $lora_r $lora_alpha $adapter_path $adapter_id
EoF
