#!/bin/bash

# Define the list of inference steps to iterate through
inference_steps=(14)

# Loop over each inference step value
for step in "${inference_steps[@]}"; do
    echo "========================================"
    echo "Running pipeline with --n_inference_step=$step"
    echo "========================================"

    # Step 1: Run fast clip and capture eval_root
    eval_root=$(python run_fast_clip.py --n_inference_step "$step" --guidance_scale 1.0 --clip_loss_coef 0.7 --fuse_coef 10 --projection_method Jacobian | grep "EVAL_ROOT=" | cut -d '=' -f2)
    echo "Captured eval_root: $eval_root"

    # Step 2: Run similarity evaluation with eval_root
    echo "Running run_eval_similarity_run.py with eval_root..."
    python run_eval_similarity_run.py --eval_root "$eval_root"

    # Step 3: Run point matching evaluation with eval_root
    echo "Running run_eval_point_matching_run.py with eval_root..."
    python run_eval_point_matching_run.py --eval_root "$eval_root"

    echo "Finished processing n_inference_step=$step"
    echo ""
done

echo "All configurations completed successfully!"