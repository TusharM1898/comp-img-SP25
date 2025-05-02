import os
import math
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from models.esrgan_model import ESRGANGenerator  # Update the path as per your model
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
VALIDATION_LR_DIR = "../data/iu_lr_images/val"  # Low-resolution validation folder
VALIDATION_HR_DIR = "../data/iu_hr_images/val"  # High-resolution validation folder
OUTPUT_DIR = "../data/ESRGAN_results/"  # Directory to save super-resolved images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Trained ESRGAN Generator
generator = ESRGANGenerator().to(DEVICE)
checkpoint = torch.load("./models/esrgan_netG_epoch2.pth", map_location=DEVICE)
generator.load_state_dict(checkpoint["model"])
generator.eval()

# Initialize Metric Logs
total_psnr = 0
total_ssim = 0
num_images = 0

# Process All Images in Validation Folder
with open(os.path.join(OUTPUT_DIR, "esrgan_metrics_log.txt"), "w") as log_file:
    log_file.write("Image\tPSNR\tSSIM\n")

    for lr_image_name in sorted(os.listdir(VALIDATION_LR_DIR)):
        lr_image_path = os.path.join(VALIDATION_LR_DIR, lr_image_name)
        hr_image_path = os.path.join(VALIDATION_HR_DIR, lr_image_name)

        if not os.path.exists(hr_image_path):
            print(f"Skipping {lr_image_name}: corresponding HR image not found.")
            continue

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

        # Save Metrics
        total_psnr += psnr_value
        total_ssim += ssim_value
        num_images += 1

        # Save Super-Resolved Image
        sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())
        output_path = os.path.join(OUTPUT_DIR, f"SR_EGAN_{lr_image_name}")
        sr_image.save(output_path)

        # Log Metrics
        log_file.write(f"{lr_image_name}\t{psnr_value:.4f}\t{ssim_value:.4f}\n")
        print(f"Processed {lr_image_name}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}")

# Calculate and Log Average Metrics
average_psnr = total_psnr / num_images if num_images > 0 else 0
average_ssim = total_ssim / num_images if num_images > 0 else 0

with open(os.path.join(OUTPUT_DIR, "esrgan_metrics_log.txt"), "a") as log_file:
    log_file.write(f"\nAverage\t{average_psnr:.4f}\t{average_ssim:.4f}\n")

print(f"Processed {num_images} images.")
print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")
