from PIL import Image
import os
from tqdm import tqdm

def resize_and_save(input_dir, output_dir_hr, output_dir_lr, hr_size=(1024, 1024), lr_size=(256, 256)):
    """
    Resize images to HR and LR pairs and save them.

    Args:
        input_dir (str): Directory containing the original images.
        output_dir_hr (str): Directory to save the HR images.
        output_dir_lr (str): Directory to save the LR images.
        hr_size (tuple): Target size for HR images (width, height).
        lr_size (tuple): Target size for LR images (width, height).
    """
    # Create directories if they don't exist
    os.makedirs(output_dir_hr, exist_ok=True)
    os.makedirs(output_dir_lr, exist_ok=True)

    # Iterate over all images in the input directory
    for file_name in tqdm(os.listdir(input_dir), desc="Processing images"):
        input_path = os.path.join(input_dir, file_name)

        # Skip if it's not an image file
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        try:
            # Open the image
            with Image.open(input_path).convert("RGB") as img:
                # Resize to HR size
                hr_image = img.resize(hr_size, Image.BICUBIC)
                # Resize to LR size
                lr_image = hr_image.resize(lr_size, Image.BICUBIC)

                # Save the images
                hr_image.save(os.path.join(output_dir_hr, file_name))
                lr_image.save(os.path.join(output_dir_lr, file_name))

        except Exception as e:
            print(f"[ERROR] Could not process file {file_name}: {e}")

# Directories
input_dir = "../data"  # Replace with the path to your IU dataset
output_dir_hr = "../data/iu_hr_images"   # Directory to save HR images
output_dir_lr = "../data/iu_lr_images"   # Directory to save LR images

# Create LR and HR pairs
resize_and_save(input_dir, output_dir_hr, output_dir_lr)
