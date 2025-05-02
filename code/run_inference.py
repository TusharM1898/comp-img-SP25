import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from model.models import Generator
from model.model_config import DEVICE, model_load_path

def load_model():
    generator = Generator(upscale_factor=4).to(DEVICE)
    generator.load_state_dict(torch.load(model_load_path)["model"])
    generator.eval()
    return generator

def run_inference(input_image_path, output_image_path):
    generator = load_model()
    input_image = Image.open(input_image_path).convert("RGB")
    lr_image = to_tensor(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sr_image = generator(lr_image).squeeze(0).cpu()
    sr_image = to_pil_image(sr_image)

    sr_image.save(output_image_path)
    print(f"[INFO] Super-resolution image saved at {output_image_path}")

# Example usage
input_image_path = "./input.png"
output_image_path = "./data/results/output.png"
run_inference(input_image_path, output_image_path)
