#!/bin/bash

lr_list=(3e-5)
mm_lr_list=(1e-4)
epoch_list=(1)
lora_r_list=(8)
adapter_list=(/home/simran/LLaVA/checkpoints/llava-v1.5-13b-llasa2-pretrain-16K-lr-1e-4-epoch-1-13b)
adapter_id=0

rm -rf /home/simran/LLaVA/slurms
mkdir /home/simran/LLaVA/slurms

for lr in "${lr_list[@]}"; do
    for epoch in "${epoch_list[@]}"; do
        for mm_lr in "${mm_lr_list[@]}"; do
            for lora_r in "${lora_r_list[@]}"; do
                for adapter in "${adapter_list[@]}"; do
                    lora_alpha=$lora_r
                    adapter_id=$((adapter_id+1))
                    bash /home/simran/LLaVA/make_run.sh $lr $mm_lr $epoch $lora_r $lora_alpha $adapter $adapter_id > /home/simran/LLaVA/slurms/run_$lr-$mm_lr-$epoch-$lora_r-$lora_alpha-$adapter_id.sh
                    sbatch /home/simran/LLaVA/slurms/run_$lr-$mm_lr-$epoch-$lora_r-$lora_alpha-$adapter_id.sh
                done
            done
        done
    done
done
