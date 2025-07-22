import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
import scipy.io as sio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra


BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR  = r"E:/restormer+volterra/data/BSDS500/images/train"
GT_TRAIN   = r"E:/restormer+volterra/data/BSDS500/ground_truth/train"
VAL_DIR    = r"E:/restormer+volterra/data/BSDS500/images/val"
GT_VAL     = r"E:/restormer+volterra/data/BSDS500/ground_truth/val"
SAVE_DIR   = r"checkpoints/restormer_volterra_bsds500_jpeg"
os.makedirs(SAVE_DIR, exist_ok=True)


resize_schedule = {0: 128, 30: 192, 60: 256}


def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


class BSDS500JPEGDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.gt_dir    = gt_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        gt_name  = os.path.splitext(img_name)[0] + ".mat"
        gt_path  = os.path.join(self.gt_dir, gt_name)

        input_img = Image.open(img_path).convert("RGB")

        gt_mat = sio.loadmat(gt_path)
        gt_img = gt_mat["groundTruth"][0, 0]["Segmentation"][0, 0]
        gt_img = gt_img.astype(np.float32) / 255.0
        gt_img = np.stack([gt_img]*3, axis=-1)
        gt_img = Image.fromarray((gt_img * 255).astype(np.uint8))

        if self.transform:
            input_img = self.transform(input_img)
            gt_img    = self.transform(gt_img)

        return input_img, gt_img


def evaluate(model, val_dir, gt_dir, transform):
    model.eval()
    input_files = sorted(os.listdir(val_dir))
    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            if not fname.endswith(".jpg"):
                continue
            input_path = os.path.join(val_dir, fname)
            gt_path    = os.path.join(gt_dir, os.path.splitext(fname)[0] + ".mat")

            input_img = transform(Image.open(input_path).convert("RGB"))
            gt_mat    = sio.loadmat(gt_path)
            gt_img    = gt_mat["groundTruth"][0, 0]["Segmentation"][0, 0]
            gt_img    = gt_img.astype(np.float32) / 255.0
            gt_img    = np.stack([gt_img]*3, axis=-1)
            gt_img    = Image.fromarray((gt_img * 255).astype(np.uint8))
            gt_img    = transform(gt_img)

            input_img = input_img.unsqueeze(0).to(DEVICE)
            gt_img    = gt_img.unsqueeze(0).to(DEVICE)

            output = model(input_img)

            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = gt_img[0].cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim


def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print(f"\n[INFO] Training JPEG Artifact Removal (BSDS500)\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        train_ds = BSDS500JPEGDataset(image_dir=TRAIN_DIR, gt_dir=GT_TRAIN, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dl)
        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f}")

        # --- Evaluate to get PSNR/SSIM for checkpoint filename ---
        test_psnr, test_ssim = evaluate(model, VAL_DIR, GT_VAL, transform)
        print(f"✅ [Epoch {epoch+1:3d}] Test  PSNR: {test_psnr:.2f} | Test  SSIM: {test_ssim:.4f}")

        # --- Save with PSNR / SSIM in filename ---
        save_name = f"epoch_{epoch+1}_ssim{test_ssim:.4f}_psnr{test_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))


if __name__ == "__main__":
    main()
