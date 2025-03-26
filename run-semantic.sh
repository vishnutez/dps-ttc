#!/bin/bash

python3 sample_condition_hyper.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config_semantic.yaml \
    --n_paths=10 \
    --batch_size=5 \
    --ref_image_idx=4 \
    --path_start_idx=0 \
    --kernel_idx=8 \
    --n_guid_images=1 \
    --sem_scale=0.1 \
    --measurement_scale=1.0 \
    --anneal_factor=4 \
    --norm_exp=1 \
    --gpu=0 ;