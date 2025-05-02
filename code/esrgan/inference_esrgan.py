# import torch
# from model.models import ESRGANGenerator
# from PIL import Image
# from torchvision.transforms import ToTensor, ToPILImage

# # Load Trained ESRGAN Generator
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = ESRGANGenerator().to(DEVICE)
# checkpoint = torch.load("./models/esrgan_netG_epoch1.pth", map_location=DEVICE)
# generator.load_state_dict(checkpoint["model"])
# generator.eval()

# # Load Low-Resolution Image
# lr_image = Image.open("../data/input_old.png").convert("RGB")
# lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(DEVICE)

# # Generate Super-Resolved Image
# with torch.no_grad():
#     sr_tensor = generator(lr_tensor)
#     sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())

# # Save and Show Image
# sr_image.save("./output/sr_image.png")
# sr_image.show()

import torch
from model.models import ESRGANGenerator
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os

# Load Trained ESRGAN Generator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = ESRGANGenerator().to(DEVICE)
checkpoint = torch.load("./models/esrgan_netG_epoch2.pth", map_location=DEVICE)
generator.load_state_dict(checkpoint["model"])
generator.eval()

# Load Low-Resolution Image
# Since X-rays are grayscale, change "RGB" to "L" if your X-rays are grayscale images.
lr_image = Image.open("../data/input_old.png").convert("RGB")  # Or "L" for grayscale
lr_tensor = ToTensor()(lr_image).unsqueeze(0).to(DEVICE)

# Generate Super-Resolved Image
with torch.no_grad():
    sr_tensor = generator(lr_tensor)
    sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)  # Clamp values to [0, 1] to avoid artifacts
    sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu())

# Save and Show Image
output_dir = "./data/results/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "esrgan_image.png")
sr_image.save(output_path)
sr_image.show()

print(f"Super-resolved image saved at {output_path}")

