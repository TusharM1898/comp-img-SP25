import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm  # For progress bar
from model.models import Generator, Discriminator
from model.model_config import DEVICE

class ChestXrayDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image

# Define paths for training and validation datasets
lr_train_dir = "./data/iu_lr_images/train"
hr_train_dir = "./data/iu_hr_images/train"
lr_val_dir = "./data/iu_lr_images/val"
hr_val_dir = "./data/iu_hr_images/val"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ChestXrayDataset(lr_train_dir, hr_train_dir, transform_lr=transform, transform_hr=transform)
val_dataset = ChestXrayDataset(lr_val_dir, hr_val_dir, transform_lr=transform, transform_hr=transform)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

generator = Generator(upscale_factor=4).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

#generator.load_state_dict(torch.load("./model/netG.pth.tar")["model"])
generator.load_state_dict(torch.load("./model/netG.pth.tar", map_location=torch.device('cpu'))["model"])
generator.train()
discriminator.train()

# Define loss functions
criterion_content = nn.MSELoss()
criterion_adversarial = nn.BCEWithLogitsLoss()

# Define optimizers
optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch [{epoch + 1}/{epochs}]")
    generator.train()
    train_loss_gen = 0.0
    train_loss_disc = 0.0

    # Training phase with progress bar
    with tqdm(train_dataloader, desc="Training", unit="batch") as tepoch:
        for lr_image, hr_image in tepoch:
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)

            # Train discriminator
            sr_image = generator(lr_image)
            real_loss = criterion_adversarial(discriminator(hr_image), torch.ones_like(discriminator(hr_image)))
            fake_loss = criterion_adversarial(discriminator(sr_image.detach()), torch.zeros_like(discriminator(sr_image)))
            disc_loss = real_loss + fake_loss

            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()

            # Train generator
            sr_image = generator(lr_image)
            content_loss = criterion_content(sr_image, hr_image)
            adversarial_loss = criterion_adversarial(discriminator(sr_image), torch.ones_like(discriminator(sr_image)))
            gen_loss = content_loss + 1e-3 * adversarial_loss

            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()

            # Update losses for progress bar
            train_loss_gen += gen_loss.item()
            train_loss_disc += disc_loss.item()
            tepoch.set_postfix(gen_loss=gen_loss.item(), disc_loss=disc_loss.item())

    train_loss_gen /= len(train_dataloader)
    train_loss_disc /= len(train_dataloader)
    print(f"Train Loss - Generator: {train_loss_gen:.4f}, Discriminator: {train_loss_disc:.4f}")

    # Validation phase
    generator.eval()
    val_loss = 0.0
    with torch.no_grad():
        with tqdm(val_dataloader, desc="Validation", unit="batch") as tepoch:
            for lr_image, hr_image in tepoch:
                lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)
                sr_image = generator(lr_image)
                val_loss += criterion_content(sr_image, hr_image).item()
                tepoch.set_postfix(val_loss=val_loss)

    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")

# Save the fine-tuned generator
torch.save({"model": generator.state_dict()}, "./model/netG_finetuned.pth.tar")
print("[INFO] Model saved as netG_finetuned.pth.tar")
