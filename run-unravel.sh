#!/bin/bash

python3 sample_condition_unravel.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config_semantic.yaml \
    --n_paths=3 \
    --batch_size=1 \
    --ref_image_idx=4 \
    --path_start_idx=0 \
    --kernel_idx=8 \
    --guid_image_idxs=4 \
    --sem_scale=2.0 \
    --measurement_scale=1.0 \
    --anneal_factor=2 \
    --norm_exp=1 \
    --gpu=0 \
    --seed=42 \
    --project;