#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=1 \
    --n_paths=5 \
    --batch_size=5 \
    --start_idx=4 \
    --path_start_idx=0 \
    --kernel_idx=8 \
    --gpu=1;
