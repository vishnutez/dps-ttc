#!/bin/bash

python3 sample_condition_batched_data.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --n_data_samples=1 \
    --n_paths=1 \
    --batch_size=1 \
    --start_idx=1 \
    --gpu=0;


# python3 sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --n_data_samples=50 \
#     --n_paths=10 \
#     --batch_size=5 \
#     --start_idx=0 \
#     --gpu=1;