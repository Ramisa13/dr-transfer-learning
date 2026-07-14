#!/usr/bin/env bash
# Task B: two-stage training (auxiliary dataset -> DeepDRiD, all layers unfrozen)
set -e
python ../src/train.py --config ../configs/config.yaml --task b \
    --stage1_data ../data/aptos --stage2_data ../data/deepdrid --unfreeze_all
