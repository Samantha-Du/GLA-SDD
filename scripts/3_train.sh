#!/bin/bash
WORKSPACE=${1:-"./workspaces/audioset_tagging"}   # Default argument.

CUDA_VISIBLE_DEVICES=0 python3 ../pytorch/main.py train \
    --workspace=$WORKSPACE \
    --data_type='balanced_train' \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type='Cnn14' \
    --loss_type='clip_bce' \
    --balanced='balanced' \
    --augmentation='mixup' \
    --batch_size=8 \
    --learning_rate=1e-4 \
    --resume_iteration=0 \
    --early_stop=80000 \
    --cuda

# Plot statistics
python3 ../utils/plot_statistics.py plot \
    --dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --select=1_aug
