#!/usr/bin/env bash
# Task D: ensemble three Task-B checkpoints with max-voting + CLAHE preprocessing
set -e
python ../src/ensemble.py \
    --checkpoints ../ckpts/model_b1.pth ../ckpts/model_b2.pth ../ckpts/model_b3.pth \
    --method max_voting --split test --out ../outputs/test_predictions_d.csv
