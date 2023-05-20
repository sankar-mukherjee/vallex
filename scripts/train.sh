#!/usr/bin/env bash

set -e

: ${DATA_DIR:=data/LibriTTS}

# train
python train.py \
    --data_dir=data/LibriTTS \
    --metadata_csv=data/LibriTTS/metadata.csv \
    --output_dir=output/LibriTTS \
    --learning_rate=0.05 \
    --decoder_dim=1024 \
    --num_heads=16 \
    --num_decoder_layers=12 \
    --num_epochs=10 \
    --batch_size=8 \
    --filter_min_duration=1 \
    --filter_max_duration=5
