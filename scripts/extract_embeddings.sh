#!/usr/bin/env bash

set -e

: ${DATA_DIR:=data/LibriTTS}

# quantization
python embeddings/quantization.py --folder $DATA_DIR
# preapare filelist
python embeddings/prepare_filelist.py --folder $DATA_DIR
