#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=10 \
    --n_particles=2;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=10 \
    --n_particles=3;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=10 \
    --n_particles=4;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_data_samples=10 \
    --n_particles=5;
    
