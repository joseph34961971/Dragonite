#!/bin/bash

# 若遇到任何錯誤即中止腳本（可依需求開啟）
#set -e

# 定義要跑的四種 interpolation_mode
modes=("ori" "0" "interpolation" "random")

for mode in "${modes[@]}"; do
  echo "=============================="
  echo "Running interpolation_mode = ${mode}"
  echo "=============================="

  # Step 1: Run fast clip and capture eval_root
  echo "Running run_fast_clip.py with interpolation_mode=${mode}..."
  eval_root=$(python run_fast_clip.py \
    --n_inference_step 10 \
    --guidance_scale 1.0 \
    --clip_loss_coef 0.7 \
    --fuse_coef 10 \
    --projection_method Jacobian \
    --interpolation_mode "${mode}" \
    | grep "EVAL_ROOT=" | cut -d '=' -f2)
  echo "  -> Captured eval_root: ${eval_root}"

  # Step 2: Run similarity evaluation
  echo "Running run_eval_similarity_run.py with eval_root=${eval_root}..."
  python run_eval_similarity_run.py --eval_root "${eval_root}"

  # Step 3: Run point matching evaluation
  echo "Running run_eval_point_matching_run.py with eval_root=${eval_root}..."
  python run_eval_point_matching_run.py --eval_root "${eval_root}"

  echo "Finished mode = ${mode}"
  echo
done

echo "All modes completed successfully!"