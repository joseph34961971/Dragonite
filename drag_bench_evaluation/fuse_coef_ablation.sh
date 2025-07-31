#!/bin/bash

set -e  # Exit on error

# Array of clip_loss_coef values to try
clip_loss_coefs=(1 10 100 500 1000 2000)

for coef in "${clip_loss_coefs[@]}"; do
  echo "==========================================="
  echo "Running with clip_loss_coef = $coef"
  echo "==========================================="

  # Step 1: Run fast clip and capture eval_root
  echo "-> Running run_fast_clip.py..."
  eval_root=$(python run_fast_clip.py \
    --n_inference_step 10 \
    --guidance_scale 1.0 \
    --clip_loss_coef "$coef" \
    --fuse_coef 10 \
    --projection_method Jacobian \
    --interpolation_mode interpolation \
    | grep "EVAL_ROOT=" | cut -d '=' -f2)
  echo "   Captured eval_root: $eval_root"

  # Step 2: Run similarity evaluation with eval_root
  echo "-> Running run_eval_similarity_run.py..."
  python run_eval_similarity_run.py --eval_root "$eval_root"

  # Step 3: Run point matching evaluation with eval_root
  echo "-> Running run_eval_point_matching_run.py..."
  python run_eval_point_matching_run.py --eval_root "$eval_root"

  echo "Completed run for clip_loss_coef = $coef"
  echo
done

echo "All coefficient runs completed successfully!"