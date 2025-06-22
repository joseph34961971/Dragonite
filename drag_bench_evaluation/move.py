import os
import shutil

# Root directory containing drag_bench_data
root_dir = "../../CLIPDrag/drag_bench_evaluation/clip_drag_res_40_iterations_80_0.7_0.01_3"
# Destination folder to collect all user_drag.png
output_dir = "./clipdrag_result"

# Create destination folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Walk through all subdirectories under root_dir
for dirpath, dirnames, filenames in os.walk(root_dir):
    if "dragged_image.png" in filenames:
        src_path = os.path.join(dirpath, "dragged_image.png")
        # Rename copied file with parent folder name to avoid overwrite
        parent_folder = os.path.basename(dirpath)
        dst_path = os.path.join(output_dir, f"{parent_folder}dragged_image.png")
        shutil.copy(src_path, dst_path)
        print(f"Copied: {src_path} -> {dst_path}")
