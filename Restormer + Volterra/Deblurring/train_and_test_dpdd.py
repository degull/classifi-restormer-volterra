# train_and_test_dpdd.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.amp import autocast, GradScaler
from models.restormer_volterra import RestormerVolterra


# -------- Dataset --------
class DPDDDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, transform=None):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.blur_files = sorted(os.listdir(blur_dir))
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_dir, self.blur_files[idx])
        sharp_path = os.path.join(self.sharp_dir, self.blur_files[idx])

        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')

        blur = self.transform(blur)
        sharp = self.transform(sharp)

        return blur, sharp


# -------- Evaluation --------
def evaluate(model, blur_dir, sharp_dir, transform, name):
    model.eval()
    files = sorted(os.listdir(blur_dir))
    total_psnr = total_ssim = 0
    count = 0

    with torch.no_grad():
        for f in files:
            blur_path = os.path.join(blur_dir, f)
            sharp_path = os.path.join(sharp_dir, f)
            if not os.path.isfile(sharp_path):
                continue
            blur = transform(Image.open(blur_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            sharp = transform(Image.open(sharp_path).convert('RGB')).unsqueeze(0).to(DEVICE)

            output = model(blur)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = sharp[0].cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"✅ [{name}] PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# -------- Main --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 50
LR = 2e-4

DPDD_TRAIN_ROOT = r"E:/DPDD/Dual_Pixel_Defocus_Deblurring/train"
DPDD_TEST_ROOT = r"E:/DPDD/Single_Image_Defocus_Deblurring/test"
SAVE_DIR = r"E:/restormer+volterra/checkpoints/dpdd"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


def run_training(train_mode):
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    train_blur = os.path.join(DPDD_TRAIN_ROOT, train_mode, 'blur')
    train_sharp = os.path.join(DPDD_TRAIN_ROOT, train_mode, 'sharp')
    train_ds = DPDDDataset(train_blur, train_sharp, transform=get_transform())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n[Train {train_mode}] Train samples: {len(train_ds)}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = tot_psnr = tot_ssim = count = 0
        loop = tqdm(train_dl, desc=f"Training {train_mode} Epoch {epoch+1}/{EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            ref_np = reference[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1
            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count
        print(f"[Epoch {epoch+1:3d}] Train Loss: {epoch_loss/len(train_dl):.6f} | Train PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{train_mode}_epoch_{epoch+1}.pth"))

    # ---- Evaluation ----
    for split in ['indoor', 'outdoor', 'combined']:
        test_blur = os.path.join(DPDD_TEST_ROOT, split, 'blur')
        test_sharp = os.path.join(DPDD_TEST_ROOT, split, 'sharp')
        evaluate(model, test_blur, test_sharp, transform=get_transform(), name=f"Test-{split.upper()} (Train-{train_mode.upper()})")


if __name__ == "__main__":
    for mode in ['indoor', 'outdoor', 'combined']:
        run_training(mode)
