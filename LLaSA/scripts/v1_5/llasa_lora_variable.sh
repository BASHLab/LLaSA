#!/bin/bash

lr=$1
mm_lr=$2
epoch=$3
lora_r=$4
lora_alpha=$5
adapter_path=$6
adapter_id=$7


deepspeed --master_port $RANDOM llava/train/train_mem.py \
    --lora_enable True --lora_r $lora_r --lora_alpha $lora_alpha --mm_projector_lr $mm_lr \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --data_path /home/simran/llasa_training_v2_raw_fixed.jsonl \
    --image_folder /home/simran/limuBERT_data/train_llasa/images/ \
    --vision_tower /home/simran/model_4.pt \
    --pretrain_mm_mlp_adapter $adapter_path/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio None \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints7/llava-v1.5-13b-llasa2-16K-lr$lr-mmlr$mm_lr-e$epoch-r$lora_r-a$lora_alpha-proj$adapter_id-lora \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps .2 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "none" \
