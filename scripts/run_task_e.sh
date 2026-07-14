#!/usr/bin/env bash
# Task E: Grad-CAM visualizations
set -e
python ../src/gradcam.py --config ../configs/config.yaml \
    --checkpoint ../ckpts/model_task_c.pth --num_samples 10 --out_dir ../assets/gradcam
