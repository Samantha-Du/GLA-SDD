#!/bin/bash
DATASET_DIR=${1:-"/media/sam/TOSHIBA/dataset/langSplit6s"}   # Default first argument.
WORKSPACE=${2:-"./workspaces/audioset_tagging"}   # Default second argument.

# Pack evaluation waveforms to a single hdf5 file
python3 ../utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path=$DATASET_DIR"/official_eval3_segments.csv" \
    --audios_dir=$DATASET_DIR"/test3" \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval3.h5"