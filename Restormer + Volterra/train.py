# train.py
# E:/MRVNet2D/Restormer + Volterra/train.py

""" import os
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
    main() """



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


# Rain100HDataset만 학습
# train_rain100h.py
# Restormer + Volterra 단일 Rain100H 학습 스크립트

""" import os
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
from re_dataset.rain100h_dataset import Rain100HDataset

# ───────────── 학습 하이퍼파라미터 ─────────────
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── 데이터·체크포인트 경로 ─────────────
RAIN100H_TRAIN_DIR = r"E:/restormer+volterra/data/rain100H/train"  # ← train split만 사용
SAVE_DIR           = r"checkpoints/restormer_volterra_rain100h"
os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────── Progressive Resize 스케줄 ─────────────
resize_schedule = {0: 128, 30: 192, 60: 256}  # epoch: shorter-side 크기

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
        # ─── 에폭별 progressive resize 적용 ───
        transform = get_transform(epoch)

        # Rain100H train split 전용 Dataset & DataLoader
        train_ds = Rain100HDataset(root_dir=RAIN100H_TRAIN_DIR, transform=transform)
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

            # ─── 지표 & 로그 ───
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
    main() """


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
from re_dataset.rain100h_dataset import Rain100HDataset

# ───────────── 설정 ─────────────
BATCH_SIZE = 2
TOTAL_EPOCHS = 100
LR = 2e-4
START_EPOCH = 34  # ✅ 이어서 학습할 epoch 번호
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAIN100H_TRAIN_DIR = r"E:/restormer+volterra/data/rain100H/train"
SAVE_DIR = r"checkpoints/restormer_volterra_rain100h"
RESUME_PATH = os.path.join(SAVE_DIR, f"epoch_{START_EPOCH - 1}.pth")

os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────── Resize 스케줄 ─────────────
resize_schedule = {0: 128, 30: 192, 60: 256}

def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

# ───────────── 학습 루프 ─────────────
def main():
    model = RestormerVolterra().to(DEVICE)

    # ✅ 이어서 학습 체크포인트 로드
    if os.path.exists(RESUME_PATH):
        print(f"🔁 이어서 학습: {RESUME_PATH}")
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
    else:
        print(f"❌ 체크포인트 없음: {RESUME_PATH}")
        return

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        transform = get_transform(epoch)
        train_ds = Rain100HDataset(root_dir=RAIN100H_TRAIN_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch:3d}] Input size: {transform.transforms[0].size} | "
              f"Samples: {len(train_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", leave=False)

        for distorted, reference in loop:
            distorted = distorted.to(DEVICE)
            reference = reference.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
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

            loop.set_postfix(loss=f"{loss.item():.4f}",
                             psnr=f"{psnr:.2f}",
                             ssim=f"{ssim:.4f}")

        print(f"📘 Epoch {epoch:3d} | Loss {epoch_loss/len(train_dl):.6f} | "
              f"PSNR {tot_psnr/count:.2f} | SSIM {tot_ssim/count:.4f}")

        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()




# 📘 Epoch  39 | Loss 0.000615 | PSNR 32.59 | SSIM 0.9357
# 📘 Epoch  40 | Loss 0.000576 | PSNR 33.00 | SSIM 0.9403
# 📘 Epoch  41 | Loss 0.000539 | PSNR 33.23 | SSIM 0.9426
# 📘 Epoch  50 | Loss 0.000346 | PSNR 35.13 | SSIM 0.9607
## 📘 Epoch  58 | Loss 0.000312 | PSNR 35.67 | SSIM 0.9680
## 📘 Epoch  59 | Loss 0.000291 | PSNR 36.00 | SSIM 0.9698

# 📘 Epoch  64 | Loss 0.000588 | PSNR 32.91 | SSIM 0.9324
# 📘 Epoch  65 | Loss 0.000554 | PSNR 33.16 | SSIM 0.9364
# 📘 Epoch  67 | Loss 0.000489 | PSNR 33.73 | SSIM 0.9451
# 📘 Epoch  68 | Loss 0.000424 | PSNR 34.29 | SSIM 0.9495



# Rain100LDataset만 학습
# train_rain100L.py
# Restormer + Volterra 단일 Rain100L 학습 스크립트
""" 
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
 """



# 이어서 학습
# train.py
# E:/MRVNet2D/Restormer + Volterra/train.py
""" 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra
from kadid_dataset import KADID10KDataset
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset

# ✅ 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로
KADID_CSV = 'E:/restormer+volterra/data/KADID10K/kadid10k.csv'
RAIN100H_DIR = 'E:/restormer+volterra/data/rain100H/train'
GOPRO_CSV = 'E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv'
SIDD_DIR = 'E:/restormer+volterra/data/SIDD'
SAVE_DIR = 'checkpoints/restormer_volterra_train_4sets'
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'epoch_98.pth')
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning
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
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device='cuda')

    resume_epoch = 0

    # ✅ 체크포인트가 존재하면 이어서 학습
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        resume_epoch = 98  # 수동 설정 (파일명 기준)

    criterion = nn.MSELoss()

    for epoch in range(resume_epoch, EPOCHS):
        transform = get_transform(epoch)

        # ✅ 데이터셋
        kadid_dataset = KADID10KDataset(csv_file=KADID_CSV, transform=transform)
        rain100h_dataset = Rain100HDataset(root_dir=RAIN100H_DIR, transform=transform)
        gopro_dataset = GoProDataset(csv_path=GOPRO_CSV, transform=transform)
        sidd_dataset = SIDD_Dataset(root_dir=SIDD_DIR, transform=transform)

        train_dataset = ConcatDataset([kadid_dataset, rain100h_dataset, gopro_dataset, sidd_dataset])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

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

            # PSNR/SSIM (batch 첫 샘플 기준)
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

        # ✅ 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()
 """
# 🏆 Best Epoch: 97 | PSNR: 28.76 | SSIM: 0.8687
