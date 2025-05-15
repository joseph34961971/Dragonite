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

# PROMPT_TEMPLATE = """You are a visual assistant. The image shows a red dot and a blue dot.
# The red dot represents the starting point, and the blue dot is the destination after dragging.
# Please describe in one sentence what changed in the image based on this drag movement.
# Focus on the semantic or meaningful visual change.
# """

PROMPT_TEMPLATE = "Generate a one line caption of what happen after the drag editing (red for starting point, blue for ending point)"

# Initialize AzureOpenAI client
client = AzureOpenAI(
    api_key="4a95e5d7c6bb49198459fea94c289fd3",                      # Required
    api_version="2024-05-01-preview",           # Required for gpt-4o
    azure_endpoint="https://openai-ntust-demo.openai.azure.com/"  # Required
)

# Your deployed Azure model name
DEPLOYMENT_NAME = "gpt-4o"  # ← Replace

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("ascii")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_gpt4o(image_path, prompt):
    image_b64 = encode_image_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_file", "file_path": image_path}  # Direct file path
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error querying GPT-4o: {str(e)}]"
    
def generate_caption(image_path):
    image_b64 = image_to_base64(image_path)
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an image analyst that describes drag edits."},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT_TEMPLATE},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ],
        max_tokens=200,
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False  
    )
    return response.choices[0].message.content.strip()

def update_metadata(folder_path):
    image_path = os.path.join(folder_path, "user_drag.png")
    metadata_path = os.path.join(folder_path, "meta_data.pkl")
    drag_prompt_path = os.path.join(folder_path, "drag_prompt.txt")

    if not os.path.exists(image_path):
        print(f"Skipping {folder_path} — 'user_drag.png' not found.")
        return

    print(f"Processing {folder_path}...")

    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}

    #print(f"Metadata loaded: {metadata}")

    gpt_response = generate_caption(image_path)
    print(f"GPT-4o response: {gpt_response}")

    # print(f"Metadata before update: {metadata}")

    # Save gpt_response to drag_prompt.txt
    with open(drag_prompt_path, "w") as f:
        f.write("gpt_response")

    metadata["drag_prompt"] = gpt_response

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    #print(f"Updated 'drag_prompt' in {folder_path}")


def create_file(folder_path):
    drag_prompt_path = os.path.join(folder_path, "drag_prompt.txt")
    with open(drag_prompt_path, "w") as f:
        f.write("")

def batch_process_all(root_dir):
    for sub in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, sub)
        if os.path.isdir(folder):
            #update_metadata(folder)
            create_file(folder)

# Example usage
if __name__ == "__main__":

    for cat in all_category:
        dataset_root = os.path.join("drag_bench_data", cat)
        print(dataset_root)
        # shared_prompt = "You are an Drag editor, given an image, please generate one-line caption to describe the object motion according to the dragging point on the image"
        batch_process_all(dataset_root)