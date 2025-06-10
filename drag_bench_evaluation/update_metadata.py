import os
import pickle
import base64
from openai import AzureOpenAI
from PIL import Image

all_category = [
    'animals',
    'art_work',
    'land_scape',
    'building_city_view',
    'building_countryside_view',
    'animals',
    'human_head',
    'human_upper_body',
    'human_full_body',
    'interior_design',
    'other_objects',
]

def update_metadata(folder_path):
    image_path = os.path.join(folder_path, "user_drag.png")
    metadata_path = os.path.join(folder_path, "meta_data.pkl")
    drag_prompt_path = os.path.join(folder_path, "drag_prompt.txt")

    if os.path.exists(drag_prompt_path):
        with open(drag_prompt_path, "r") as f:
            drag_prompt = f.read().strip()
    else:
        print("empty!!")
        drag_prompt = ""

    print(drag_prompt)

    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}

    print("metadata:", metadata)

    metadata["drag_prompt"] = drag_prompt

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def batch_process_all(root_dir):
    for sub in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, sub)
        print(folder)
        if os.path.isdir(folder):
            #update_metadata(folder)
            update_metadata(folder)

# Example usage
if __name__ == "__main__":

    for cat in all_category:
        dataset_root = os.path.join("drag_bench_data_gen", cat)
        print(dataset_root)
        # shared_prompt = "You are an Drag editor, given an image, please generate one-line caption to describe the object motion according to the dragging point on the image"
        batch_process_all(dataset_root)