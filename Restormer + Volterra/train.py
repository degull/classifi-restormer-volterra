# train.py
# E:/MRVNet2D/Restormer + Volterra/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # ✅ 최신 버전 사용
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra
from kadid_dataset import KADID10KDataset
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset

# ✅ 학습 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
KADID_CSV = 'E:/restormer+volterra/data/KADID10K/kadid10k.csv'
RAIN100H_DIR = 'E:/restormer+volterra/data/rain100H/train'
GOPRO_CSV = 'E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv'
SIDD_DIR = 'E:/restormer+volterra/data/SIDD'

SAVE_DIR = 'checkpoints/restormer_volterra_train_4sets'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning 스케줄
resize_schedule = {
    0: 128,
    30: 192,
    60: 256
}

def get_transform(epoch):
    size = 256
    for key in sorted(resize_schedule.keys()):
        if epoch >= key:
            size = resize_schedule[key]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device='cuda')

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        # ✅ 데이터셋 로드
        kadid_dataset = KADID10KDataset(csv_file=KADID_CSV, transform=transform)
        rain100h_dataset = Rain100HDataset(root_dir=RAIN100H_DIR, transform=transform)
        gopro_dataset = GoProDataset(csv_path=GOPRO_CSV, transform=transform)
        sidd_dataset = SIDD_Dataset(root_dir=SIDD_DIR, transform=transform)

        # ✅ 데이터셋 통합
        train_dataset = ConcatDataset([
            kadid_dataset,
            rain100h_dataset,
            gopro_dataset,
            sidd_dataset
        ])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1}] Input resolution: {transform.transforms[0].size}, Total samples: {len(train_dataset)}")

        model.train()
        epoch_loss = 0
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # ✅ PSNR / SSIM 계산 (batch 내 첫 샘플 기준)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            ref_np = reference[0].detach().cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, channel_axis=2, data_range=1.0, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            loop.set_postfix(loss=loss.item(), psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.6f} | "
              f"Avg PSNR: {total_psnr / count:.2f} | Avg SSIM: {total_ssim / count:.4f}")

        # ✅ 모델 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()



"""
입력 해상도: 256×256×C

Encoder 흐름:

L1: 256×256×C

L2: 128×128×2C

L3: 64×64×4C

L4: 32×32×8C

Decoder 흐름:

L3: 64×64×4C

L2: 128×128×2C

L1: 256×256×C

최종 출력: 256×256×C

즉, **입력과 출력은 동일 해상도 (256×256)**이며, encoder는 4단계 downsampling, decoder는 3단계 upsampling 구조입니다.

"""