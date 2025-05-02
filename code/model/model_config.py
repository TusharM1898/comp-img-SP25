import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
upscale_factor = 4
model_load_path = "./model/netG.pth.tar"
