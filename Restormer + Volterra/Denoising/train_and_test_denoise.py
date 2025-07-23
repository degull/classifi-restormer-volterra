import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.amp import autocast, GradScaler
from restormer_volterra import RestormerVolterra


class SIDD_Dataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs.iloc[idx]
        noisy = Image.open(noisy_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        return self.transform(noisy), self.transform(gt)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_sidd"
os.makedirs(SAVE_DIR, exist_ok=True)

SIDD_CSV = r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv"

resize_schedule = {0: 128, 30: 192, 60: 256}


def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def evaluate_sidd(model, dataloader):
    model.eval()
    total_psnr = total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for noisy, gt in dataloader:
            noisy = noisy.to(DEVICE)
            gt = gt.to(DEVICE)
            output = model(noisy)
            gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"✅ SIDD VAL PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def main():
    # CSV 하나 불러옴
    pairs = pd.read_csv(SIDD_CSV)

    # 여기서 train / val split
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"\n[INFO] Training on 80% / Validating on 20%\n")

    history = {
        'train_psnr': [], 'train_ssim': [],
        'val_psnr': [], 'val_ssim': [],
    }

    val_ds = SIDD_Dataset(pairs=val_pairs, transform=get_transform(0))
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = SIDD_Dataset(pairs=train_pairs, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)

        for noisy, gt in loop:
            noisy = noisy.to(DEVICE)
            gt = gt.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(noisy)
                loss = criterion(output, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count
        avg_loss = epoch_loss / len(train_dl)

        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} | Train SSIM: {avg_ssim:.4f}")

        val_psnr, val_ssim = evaluate_sidd(model, val_dl)

        history['train_psnr'].append(avg_psnr)
        history['train_ssim'].append(avg_ssim)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    print("\n📊 [전체 Epoch별 요약 PSNR/SSIM]\n")
    header = f"{'Ep':>3} | {'Train':>12} | {'VAL':>12}"
    print(header)
    print("-" * len(header))
    for i in range(EPOCHS):
        print(f"{i+1:3d} | "
              f"{history['train_psnr'][i]:.2f}/{history['train_ssim'][i]:.3f} | "
              f"{history['val_psnr'][i]:.2f}/{history['val_ssim'][i]:.3f}")


if __name__ == "__main__":
    main()

# 📊 [전체 Epoch별 요약 PSNR/SSIM]
# 
#  Ep |        Train |          VAL
# ---------------------------------
#   1 | 25.82/0.653 | 29.63/0.853
#   2 | 30.45/0.874 | 30.70/0.905
#   3 | 33.59/0.935 | 33.22/0.937
#   4 | 34.87/0.949 | 33.52/0.957
#   5 | 34.67/0.953 | 33.73/0.951
#   6 | 33.74/0.943 | 31.84/0.956
#   7 | 36.28/0.968 | 36.37/0.971
#   8 | 37.55/0.977 | 36.83/0.972
#   9 | 36.40/0.967 | 37.29/0.969
#  10 | 38.12/0.976 | 37.37/0.977
#  11 | 38.74/0.981 | 37.93/0.979
#  12 | 37.45/0.975 | 36.28/0.979
#  13 | 37.82/0.979 | 38.73/0.981
#  14 | 38.01/0.981 | 37.88/0.981
#  15 | 39.34/0.983 | 38.82/0.982
#  16 | 38.53/0.982 | 36.55/0.981
#  17 | 39.15/0.985 | 38.66/0.984
#  18 | 38.79/0.985 | 34.22/0.979
#  19 | 38.31/0.976 | 37.16/0.974
#  20 | 38.41/0.980 | 37.02/0.981
#  21 | 36.42/0.975 | 32.18/0.958
#  22 | 38.04/0.981 | 38.82/0.982
#  23 | 38.65/0.982 | 37.12/0.981
#  24 | 35.48/0.969 | 34.84/0.950
#  25 | 37.00/0.971 | 40.01/0.982
#  26 | 39.92/0.986 | 39.46/0.985
#  27 | 38.52/0.981 | 38.45/0.981
#  28 | 38.40/0.980 | 38.80/0.985
#  29 | 40.21/0.986 | 40.63/0.985
#  30 | 40.85/0.989 | 41.21/0.986
#  31 | 39.14/0.978 | 37.99/0.978
#  32 | 39.90/0.982 | 37.64/0.983
#  33 | 39.65/0.982 | 39.42/0.983
#  34 | 41.37/0.984 | 39.37/0.987
#  35 | 41.50/0.986 | 40.02/0.987
#  36 | 39.76/0.980 | 39.53/0.983
#  37 | 40.64/0.983 | 38.74/0.985
#  38 | 40.72/0.983 | 38.87/0.987
#  39 | 41.04/0.985 | 40.94/0.988
#  40 | 40.82/0.986 | 36.26/0.978
#  41 | 39.18/0.977 | 39.45/0.984
#  42 | 38.04/0.973 | 40.06/0.984
#  43 | 37.31/0.972 | 38.67/0.977
#  44 | 41.23/0.982 | 40.20/0.987
#  45 | 42.31/0.985 | 39.81/0.988
#  46 | 41.81/0.985 | 40.63/0.987
#  47 | 41.68/0.986 | 39.14/0.984
#  48 | 40.35/0.981 | 40.29/0.988
#  49 | 41.27/0.984 | 39.68/0.988
#  50 | 39.51/0.981 | 38.23/0.986
#  51 | 41.27/0.984 | 39.02/0.987
#  52 | 41.55/0.986 | 41.25/0.989
#  53 | 41.19/0.985 | 40.71/0.989
#  54 | 41.21/0.986 | 36.26/0.986
#  55 | 37.59/0.972 | 38.61/0.984
#  56 | 39.26/0.980 | 37.70/0.984
#  57 | 41.47/0.985 | 40.67/0.988
#  58 | 41.92/0.986 | 41.22/0.989
#  59 | 42.08/0.987 | 40.72/0.989
#  60 | 42.53/0.987 | 39.99/0.989
#  61 | 41.62/0.980 | 41.20/0.989
#  62 | 41.81/0.981 | 41.41/0.989
#  63 | 41.91/0.983 | 40.34/0.989
#  64 | 41.21/0.981 | 40.55/0.989
#  65 | 41.11/0.983 | 36.21/0.985
#  66 | 39.48/0.980 | 40.82/0.986
#  67 | 39.73/0.979 | 37.02/0.986
#  68 | 39.37/0.977 | 37.61/0.974
#  69 | 39.65/0.973 | 40.08/0.986
#  70 | 40.44/0.980 | 39.46/0.987
#  71 | 41.60/0.982 | 40.62/0.988
#  72 | 41.59/0.981 | 41.21/0.990
#  73 | 42.79/0.983 | 41.35/0.990
#  74 | 42.57/0.985 | 41.45/0.990
#  75 | 42.59/0.985 | 40.89/0.989
#  76 | 42.25/0.986 | 40.98/0.990
#  77 | 40.71/0.984 | 38.91/0.985
#  78 | 40.95/0.978 | 41.45/0.989
#  79 | 42.16/0.983 | 39.82/0.987
#  80 | 40.39/0.979 | 40.37/0.988
#  81 | 40.88/0.984 | 39.44/0.988
#  82 | 40.95/0.983 | 41.02/0.988
#  83 | 41.81/0.982 | 41.93/0.990
#  84 | 42.62/0.985 | 40.30/0.987
#  85 | 41.73/0.983 | 39.77/0.988
#  86 | 40.95/0.982 | 41.24/0.988
#  87 | 41.19/0.982 | 39.73/0.988
#  88 | 42.49/0.985 | 39.51/0.988
#  89 | 41.15/0.982 | 41.00/0.988
#  90 | 42.72/0.986 | 41.74/0.990
#  91 | 42.66/0.983 | 40.70/0.988
#  92 | 41.12/0.983 | 42.09/0.990
#  93 | 42.81/0.986 | 41.47/0.989
#  94 | 41.50/0.984 | 40.51/0.987
#  95 | 42.54/0.986 | 41.98/0.990
#  96 | 42.54/0.987 | 41.17/0.989
#  97 | 43.02/0.985 | 42.07/0.991
#  98 | 42.58/0.985 | 41.66/0.988
#  99 | 42.82/0.985 | 39.97/0.989
# 100 | 39.97/0.982 | 36.59/0.985