import os
import shutil
import re

def extract_filenames_from_dir(directory):
    """Extract base names from filenames that contain '_combined' or '_combined_image'."""
    extracted_names = set()
    for filename in os.listdir(directory):
        name, _ = os.path.splitext(filename)
        match = re.match(r'^(.*?)_combined(?:_image)?$', name)
        if match:
            extracted_names.add(match.group(1))
    return extracted_names

def copy_matching_dirs(source_root_dir, name_set, output_dir):
    """Search through nested category dirs for matching names, and copy them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for category in os.listdir(source_root_dir):
        category_path = os.path.join(source_root_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        for subdir in os.listdir(category_path):
            if subdir in name_set:
                src = os.path.join(category_path, subdir)
                dst = os.path.join(output_dir, subdir)
                if os.path.isdir(src):
                    print(f"Copying: {src} -> {dst}")
                    shutil.copytree(src, dst, dirs_exist_ok=True)

# ==== CONFIGURATION ====
combined_dir = "./quality_results"       # Directory with '_combined' files
nested_dir_root = "./DragBench"     # Root directory with category subfolders
output_dir = "./output"           # Where matching dirs will be copied

# ==== EXECUTION ====
name_list = extract_filenames_from_dir(combined_dir)
copy_matching_dirs(nested_dir_root, name_list, output_dir)