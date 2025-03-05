#!/bin/bash

python3 sample_condition_batched_data.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=5 \
    --n_paths=1 \
    --batch_size=5 \
    --start_idx=40 \
    --path_start_idx=0 \
    --kernel_idx=8 \
    --gpu=0;