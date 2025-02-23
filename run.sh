#!/bin/bash

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --n_data_samples=2 \
    --n_paths=10 \
    --batch_size=5;
