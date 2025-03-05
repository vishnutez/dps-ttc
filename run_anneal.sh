#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config_anneal.yaml \
    --n_data_samples=1 \
    --n_paths=3 \
    --batch_size=3 \
    --start_idx=40 \
    --path_start_idx=3 \
    --kernel_idx=37 \
    --anneal_amp=1.5 \
    --gpu=0;