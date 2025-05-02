import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir_train, output_dir_val, split_ratio=0.8):
    """
    Split dataset into training and validation sets.

    Args:
        input_dir (str): Directory containing images to split.
        output_dir_train (str): Directory to save training images.
        output_dir_val (str): Directory to save validation images.
        split_ratio (float): Proportion of images to use for training. Default is 0.8.
    """
    # Create train and val directories
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)

    # Get all image files
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Split into train and validation sets
    train_files, val_files = train_test_split(images, train_size=split_ratio, random_state=42)

    # Move files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir_train, file))
    for file in val_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir_val, file))

    print(f"[INFO] Dataset split complete: {len(train_files)} train, {len(val_files)} val")

# Directories for HR images
hr_input_dir = "./data/iu_hr_images"
hr_train_dir = "./data/iu_hr_images/train"
hr_val_dir = "./data/iu_hr_images/val"

# Directories for LR images
lr_input_dir = "./data/iu_lr_images"
lr_train_dir = "./data/iu_lr_images/train"
lr_val_dir = "./data/iu_lr_images/val"

# Split HR and LR datasets
split_dataset(hr_input_dir, hr_train_dir, hr_val_dir, split_ratio=0.8)
split_dataset(lr_input_dir, lr_train_dir, lr_val_dir, split_ratio=0.8)