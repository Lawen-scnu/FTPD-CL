#!/bin/bash
module purge
#module load anaconda
module load miniforge3/24.1
module load compilers/cuda/11.6
module load cudnn/8.4.0.27_cuda11.x
module load compilers/gcc/9.3.0
source activate scnulm

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model /deepseek-vl-7b-chat\
    --device_map auto \
    --train_type lora \
    --dataset '/train.jsonl' \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
