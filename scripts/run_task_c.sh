#!/usr/bin/env bash
# Task C: self-attention augmented ResNet
set -e
python ../src/train.py --config ../configs/config.yaml --task c --use_attention
