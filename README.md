# RPA-SDD: Rhythm and Pitch Aware Dual-Branch Network for Songs Deepfake Detection

This repo contains code for our paper: **RPA-SDD: Rhythm and Pitch Aware Dual-Branch Network for Songs Deepfake Detection**. Our MDS dataset are available at https://drive.google.com/file/d/1rFsvMYihVtk81uFbL7UpyUEs-qBgsX6H/view?usp=drive_link

## Environments
The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```

## Train RPA-SDD from scratch
You can train RPA-SDD from scratch as follows.

## 1. Pack waveforms into hdf5 files
The [scripts/2_pack_waveforms_to_hdf5s.sh](scripts/2_pack_waveforms_to_hdf5s.sh) script is used for packing all raw waveforms into large hdf5 files for speed up training. The packed files looks like:

<pre>
workspace
└── hdf5s
     ├── indexs
          ├── train.h5
          ├── train_demucs.h5
          └── eval1.h5
          └── eval2.h5
          └── eval3.h5
     └── waveforms (1.1 TB)
          ├── train.h5
          ├── train_demucs.h5
          └── eval1.h5
          └── eval2.h5
          └── eval3.h5
</pre>



## 2. Create training indexes
The [scripts/3_create_training_indexes.sh](scripts/3_create_training_indexes.sh) is used for creating training indexes. Those indexes are used for sampling mini-batches.

## 3. Train
The [scripts/4_train.sh](scripts/4_train.sh) script contains training, saving checkpoints, and evaluation.

```
WORKSPACE="your_workspace"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
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
  --learning_rate=1e-3 \
  --resume_iteration=0 \
  --early_stop=1000000 \
  --cuda
```

## Results
The RPA-SDD model is trained on a single card NVIDIA GeForce RTX 2080 Ti.  The training takes around 7-9 hours. 

```
Validate bal eer: 0.524
Validate test eer: 0.523
     Dump statistics to ./workspaces/audioset_tagging/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=balanced_train/Cnn14/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=8/statistics.pkl
      Dump statistics to ./workspaces/audioset_tagging/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=balanced_train/Cnn14/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=8/statistics_2023-12-30_07-10-10.pkl

iteration: 0, train time: 40.052 s, validate time: 35.265 s
------------------------------------
...
------------------------------------
Validate bal mAP: 0.035
Validate test mAP: 0.023
    Dump statistics to ./workspaces/audioset_tagging/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=balanced_train/Cnn14/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=8/statistics.pkl
    Dump statistics to ./workspaces/audioset_tagging/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=balanced_train/Cnn14/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=8/statistics_2023-12-30_15-42-10.pkl

iteration: 78000, train time: 41.070 s, validate time: 30.770 s
------------------------------------
...
```

