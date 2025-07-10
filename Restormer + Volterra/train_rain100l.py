# Rain100LDataset만 학습
# train_rain100L.py
# Restormer + Volterra 단일 Rain100L 학습 스크립트

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset  # ✅ Rain100L용 Dataset import

# ───────────── 학습 하이퍼파라미터 ─────────────
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── 데이터·체크포인트 경로 ─────────────
RAIN100L_TRAIN_DIR = r"E:/restormer+volterra/data/rain100L/train"  # ← Rain100L의 train split만 사용
SAVE_DIR           = r"checkpoints/restormer_volterra_rain100l"
os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────── Progressive Resize 스케줄 ─────────────
resize_schedule = {0: 128, 30: 192, 60: 256}

def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

# ───────────── 메인 학습 루프 ─────────────
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        # ✅ Rain100L train split 전용 Dataset & DataLoader
        train_ds = Rain100LDataset(root_dir=RAIN100L_TRAIN_DIR, transform=transform)
        train_dl = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} "
              f"| Samples: {len(train_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = model(distorted)
                loss   = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            ref_np = reference[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr   = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim   = compute_ssim(ref_np, out_np, data_range=1.0,
                                  channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count    += 1

            loop.set_postfix(loss=f"{loss.item():.4f}",
                             psnr=f"{psnr:.2f}",
                             ssim=f"{ssim:.3f}")

        print(f"Epoch {epoch+1:3d} | "
              f"Loss {epoch_loss/len(train_dl):.6f} | "
              f"PSNR {tot_psnr/count:.2f} | "
              f"SSIM {tot_ssim/count:.4f}")

        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()
