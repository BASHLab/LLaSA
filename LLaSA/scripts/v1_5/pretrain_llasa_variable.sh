#!/bin/bash

lr=$1
epoch=$2


deepspeed  --master_port $RANDOM llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --version plain \
    --data_path /home/simran/llasa_training_v2_raw_fixed.jsonl \
    --image_folder /home/simran/limuBERT_data/train_llasa/images/ \
    --vision_tower /home/simran/model_4.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-llasa2-pretrain-16K-lr-$lr-epoch-$epoch-13b \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --report_to "none"
