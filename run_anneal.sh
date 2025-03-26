#!/bin/bash

python3 sample_condition_batched_ttc.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --n_data_samples=1 \
    --n_paths=8 \
    --batch_size=8 \
    --start_idx=40 \
    --path_start_idx=0 \
    --kernel_idx=37 \
    --anneal_amp=1.5 \
    --ref_image_idxs=3 \
    --gpu=0;