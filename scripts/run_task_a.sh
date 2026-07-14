#!/usr/bin/env bash
# Task A: baseline fine-tuning (frozen backbone, train head only)
set -e
python ../src/train.py --config ../configs/config.yaml --task a --freeze_backbone
