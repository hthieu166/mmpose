#!/bin/bash
DATA_ROOT=/mnt/data0-nfs/shared-datasets/human36m
# 10 fps
python preprocess_h36m.py \
    --metadata  $DATA_ROOT/metadata.xml \
    --extracted $DATA_ROOT/extracted \
    --processed $DATA_ROOT/processed \
    --sample-rate 5
# 50 fps
python preprocess_h36m.py \
    --metadata  $DATA_ROOT/metadata.xml \
    --extracted $DATA_ROOT/extracted \
    --processed $DATA_ROOT/processed \
    --sample-rate 1