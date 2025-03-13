#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config_semantic.yaml \
    --n_data_samples=1 \
    --n_paths=1 \
    --batch_size=1 \
    --start_idx=4 \
    --path_start_idx=7 \
    --kernel_idx=8 \
    --guid_image_idx=3 \
    --gpu=0;