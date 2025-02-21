#!/bin/bash

# python3 sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/diffusion_config.yaml \
#     --task_config=configs/inpainting_config.yaml \
#     --n_data_samples=2 \
#     --n_particles=2;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --n_data_samples=2 \
    --n_particles=2 \
    --start_idx=0;


# python3 sample_condition.py \
#     --model_config=configs/model_config.yaml \
#     --diffusion_config=configs/diffusion_config.yaml \
#     --task_config=configs/gaussian_deblur_config.yaml \
#     --n_data_samples=2 \
#     --n_particles=2 \
#     --start_idx=4 \
#     --path_start_idx=2;