import os
from PIL import Image

ori_dir = "./drag_bench_data"
result_dir = "./results/fast_clip_inter_nolora_kvcopy_inverse10_80_0.7_0.01_3_0.7_10.0_Jacobian2025-06-17_03:27:52"

def combine_matching_images(ori_dir, result_dir, output_folder):
    # Loop through all subdirectories in folder1
    for categories in os.listdir(result_dir):
        result_cat_dir = os.path.join(result_dir, categories)
        ori_cat_dir = os.path.join(ori_dir, categories)
        # print(ori_cat_dir)
        # print(result_cat_dir)

        for subfolder in os.listdir(result_cat_dir):
            subfolder_path1 = os.path.join(ori_cat_dir, subfolder)
            subfolder_path2 = os.path.join(result_cat_dir, subfolder)
            # print(subfolder_path1)
            # print(subfolder_path2)
            print(os.path.isdir(subfolder_path1))
            print(subfolder_path1)
            # print(subfolder_path1)

            if os.path.isdir(subfolder_path1) and os.path.isdir(subfolder_path2):
                img1_path = os.path.join(subfolder_path1, "user_drag.png")
                img2_path = os.path.join(subfolder_path2, "dragged_image.png")
                # output_path = os.path.join(subfolder_path2, "combined_image.png")
                print(img1_path)
                print(img2_path)

                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    print("yes")
                    # Open and align height if needed
                    img1 = Image.open(img1_path)
                    img2 = Image.open(img2_path)

                    if img1.height != img2.height:
                        img2 = img2.resize(
                            (int(img2.width * img1.height / img2.height), img1.height)
                        )

                    # Create new image
                    combined = Image.new("RGB", (img1.width + img2.width, img1.height))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))

                    # Save
                    output_path = os.path.join(output_folder, f"{subfolder}_combined.jpg")
                    combined.save(output_path)
                    print(f"Saved: {output_path}")
                else:
                    print(f"Skipped: Missing images in {subfolder}")

# Example usage
combine_matching_images(
    ori_dir=ori_dir,
    result_dir=result_dir,
    output_folder=result_dir
)