import os
from PIL import Image

# Set directory paths
dir1 = './dragonite_fastdrag'
dir2 = './clipdrag_result'
output_dir = 'dragonite_fastdrag_clipdrag'
os.makedirs(output_dir, exist_ok=True)

# Helper: Extract base timestamp key
def extract_key(filename):
    return filename.split('draged_image')[0] if '_user_drag' in filename else filename.split('_merged')[0]

# Build mapping of key -> filename
dir1_map = {extract_key(f): f for f in os.listdir(dir1) if f.endswith('.jpg')}
dir2_map = {extract_key(f): f for f in os.listdir(dir2) if f.endswith('.png')}

print(dir1_map)
print("aaaaaaaaaaaaaaa")
print(dir2_map)

# Match and process
matched_keys = dir1_map.keys() & dir2_map.keys()

for key in matched_keys:
    dir1_path = os.path.join(dir1, dir1_map[key])
    dir2_path = os.path.join(dir2, dir2_map[key])

    try:
        img1 = Image.open(dir1_path)
        img2 = Image.open(dir2_path)

        # Resize to same height
        if img1.height != img2.height:
            ratio = img1.height / img2.height
            img2 = img2.resize((int(img2.width * ratio), img1.height))

        # Combine side by side
        combined_img = Image.new("RGB", (img1.width + img2.width, img1.height))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

        # Save with shared key name
        combined_img.save(os.path.join(output_dir, f"{key}_merged.jpg"))
        print(f"✔️ Combined {key}")
    except Exception as e:
        print(f"❌ Failed for {key}: {e}")