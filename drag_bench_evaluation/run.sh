#!/bin/bash

# # Exit on error
# set -e

# Step 1: Run fast clip and capture eval_root
echo "Running run_fast_clip.py..."
eval_root=$(python run_fast_clip.py --guidance_scale 1.0 --clip_loss_coef 0.7 --fuse_coef 100 --projection_method Naive | grep "EVAL_ROOT=" | cut -d '=' -f2)
echo "Captured eval_root: $eval_root"

# Step 2: Run similarity evaluation with eval_root
echo "Running run_eval_similarity_run.py with eval_root..."
python run_eval_similarity_run.py --eval_root "$eval_root"

# Step 3: Run point matching evaluation with eval_root
echo "Running run_eval_point_matching_run.py with eval_root..."
python run_eval_point_matching_run.py --eval_root "$eval_root"

echo "All scripts executed successfully!"