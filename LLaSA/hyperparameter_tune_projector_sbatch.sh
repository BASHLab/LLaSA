#!/bin/bash

lr_list=(1e-4)
epoch_list=(1)

rm -rf /home/simran/LLaVA/slurms_pre
mkdir /home/simran/LLaVA/slurms_pre

for lr in "${lr_list[@]}"; do
    for epoch in "${epoch_list[@]}"; do
        bash /home/simran/LLaVA/make_run_pre.sh $lr $epoch > /home/simran/LLaVA/slurms_pre/run_$lr-$epoch.sh
        sbatch /home/simran/LLaVA/slurms_pre/run_$lr-$epoch.sh
    done
done
