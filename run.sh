#!/bin/bash


python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_particles=2 \
    --resample_every_steps=20 \
    --potential_type='mean' \
    --rs_temp=2 ;
