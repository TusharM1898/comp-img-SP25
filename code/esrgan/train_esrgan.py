import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model.models import ESRGANGenerator, ESRGANDiscriminator
from model.custom_loss import ESRGANLoss
from torchvision.transforms import ToTensor
from PIL import Image

# Dataset class
class ChestXrayDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform if transform else ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        return self.transform(lr_image), self.transform(hr_image)


def fine_tune_esrgan():
    # Configuration
    LR_TRAIN_DIR = "../data/iu_lr_images/train"
    HR_TRAIN_DIR = "../data/iu_hr_images/train"
    LR_VAL_DIR = "../data/iu_lr_images/val"
    HR_VAL_DIR = "../data/iu_hr_images/val"
    UPSCALE_FACTOR = 4
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    train_set = ChestXrayDataset(LR_TRAIN_DIR, HR_TRAIN_DIR)
    val_set = ChestXrayDataset(LR_VAL_DIR, HR_VAL_DIR)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Load Models
    generator = ESRGANGenerator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    discriminator = ESRGANDiscriminator().to(DEVICE)

    # Loss and Optimizers
    loss_fn = ESRGANLoss().to(DEVICE)
    optimizerG = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    optimizerD = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

    # Mixed Precision Training
    scalerG = torch.cuda.amp.GradScaler()
    scalerD = torch.cuda.amp.GradScaler()

    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}] Training")
        for lr_image, hr_image in train_bar:
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                sr_image = generator(lr_image).detach()
                real_out = discriminator(hr_image).mean()
                fake_out = discriminator(sr_image).mean()
                d_loss = torch.mean(1 - real_out + fake_out)

            optimizerD.zero_grad()
            scalerD.scale(d_loss).backward()
            scalerD.step(optimizerD)
            scalerD.update()

            # Train Generator
            with torch.cuda.amp.autocast():
                sr_image = generator(lr_image)
                fake_out = discriminator(sr_image).mean()
                g_loss = loss_fn(sr_image, hr_image, real_out, fake_out)

            optimizerG.zero_grad()
            scalerG.scale(g_loss).backward()
            scalerG.step(optimizerG)
            scalerG.update()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")

        # Save Models
        os.makedirs("./models", exist_ok=True)
        torch.save({"model": generator.state_dict()}, f"./models/esrgan_netG_epoch{epoch}.pth")
        torch.save({"model": discriminator.state_dict()}, f"./models/esrgan_netD_epoch{epoch}.pth")


if __name__ == "__main__":
    fine_tune_esrgan()
