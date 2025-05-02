import os
import math
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from models.srgan_model import Generator  # Update the path as per your model
from skimage.metrics import structural_similarity as ssim_metric

# PSNR Calculation Function
def calculate_psnr(sr_image, hr_image):
    """Calculate PSNR between SR and HR images."""
    mse = torch.mean((sr_image - hr_image) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

# SSIM Calculation Function
def calculate_ssim(sr_image, hr_image, win_size=3):
    """Calculate SSIM between SR and HR images."""
    sr_image_np = sr_image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    hr_image_np = hr_image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

    # Ensure the window size is valid for the image dimensions
    min_dim = min(sr_image_np.shape[:2])
    adjusted_win_size = min(win_size, min_dim)

    ssim_value = ssim_metric(
        sr_image_np,
        hr_image_np,
        multichannel=True,
        data_range=1.0,
        win_size=adjusted_win_size,
    )
    return ssim_value

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./output/"  # Directory to save super-resolved image
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Trained SRGAN Generator
generator = Generator().to(DEVICE)
checkpoint = torch.load("./models/SRGAN_netG_epoch7.pth", map_location=DEVICE)
generator.load_state_dict(checkpoint["model"])
generator.eval()

# Process a Single Image
def process_single_image(lr_image_path, hr_image_path):
    # Check if the HR image exists
    if not os.path.exists(hr_image_path):
        print(f"High-resolution image not found: {hr_image_path}")
        return

    # Load Low-Resolution and High-Resolution Images
    lr_image = Image.open(lr_image_path).convert("RGB")  # Use "L" for grayscale if needed
    hr_image = Image.open(hr_image_path).convert("RGB")

    lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(DEVICE)
    hr_tensor = ToTensor()(hr_image).unsqueeze(0).to(DEVICE)

    # Generate Super-Resolved Image
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)  # Clamp values to [0, 1]

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(sr_tensor.squeeze(0), hr_tensor.squeeze(0))
    ssim_value = calculate_ssim(sr_tensor.squeeze(0), hr_tensor.squeeze(0))

    # Save Super-Resolved Image
    sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
    output_path = os.path.join(OUTPUT_DIR, f"SRGAN_{os.path.basename(lr_image_path)}")
    sr_image.save(output_path)

    # Log Metrics
    print(f"Processed {os.path.basename(lr_image_path)}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}")
    print(f"Super-resolved image saved at: {output_path}")

# Example Usage
lr_image_path = "/Users/tusharmuley/Desktop/JHU/Fall24/MLSP/Project/Chest_Xray_IU/data/iu_lr_images/val/474_IM-2101-1001.png"  # Path to your LR image
hr_image_path = "/Users/tusharmuley/Desktop/JHU/Fall24/MLSP/Project/Chest_Xray_IU/data/iu_hr_images/val/474_IM-2101-1001.png"  # Path to the corresponding HR image

process_single_image(lr_image_path, hr_image_path)
