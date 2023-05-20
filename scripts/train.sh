#!/usr/bin/env bash

set -e

: ${DATA_DIR:=data/LibriTTS}

# train
python train.py \
    --data_dir=$DATA_DIR \
    --metadata_csv_train=$DATA_DIR/metadata_train.csv \
    --metadata_csv_val=$DATA_DIR/metadata_val.csv \
    --output_dir=$DATA_DIR \
    --learning_rate=0.05 \
    --decoder_dim=1024 \
    --num_heads=16 \
    --num_decoder_layers=12 \
    --num_epochs=100 \
    --batch_size=8 \
    --filter_min_duration=1 \
    --filter_max_duration=10
