# Train : Deraining + Deblurring + Denoising + Desnowing + JPEG
# Test : Deraining + Deblurring + Denoising + Desnowing + JPEG
# 01_train_all_tasks.py

""" import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.amp import autocast, GradScaler

from models.restormer_volterra import RestormerVolterra

# ---------------------- Dataset Templates ----------------------
class PairedFolderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        self.input_paths = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])
        self.target_paths = sorted([
            os.path.join(target_dir, f) for f in os.listdir(target_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)

class PairedCSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)

# ---------------------- Config ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
TOTAL_EPOCHS = 100
LR = 2e-4
SAVE_DIR = r"E:/restormer+volterra/checkpoints/#01_all_tasks"
os.makedirs(SAVE_DIR, exist_ok=True)

START_EPOCH = 0  # 새로 학습 시작
resize_schedule = {0: 128, 30: 192, 60: 256}

def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

# ---------------------- Dataset Loaders ----------------------
def get_all_datasets(transform):
    datasets = []

    print("\n📦 Preparing datasets...")

    rain100h = PairedFolderDataset(
        input_dir=r"E:/restormer+volterra/data/rain100H/train/rain",
        target_dir=r"E:/restormer+volterra/data/rain100H/train/norain",
        transform=transform
    )
    print(f"Rain100H      : {len(rain100h)} samples")
    datasets.append(rain100h)

    rain100l = PairedFolderDataset(
        input_dir=r"E:/restormer+volterra/data/rain100L/train/rain",
        target_dir=r"E:/restormer+volterra/data/rain100L/train/norain",
        transform=transform
    )
    print(f"Rain100L      : {len(rain100l)} samples")
    datasets.append(rain100l)

    gopro = PairedCSVDataset(
        csv_path=r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
        transform=transform
    )
    print(f"GoPro         : {len(gopro)} samples")
    datasets.append(gopro)

    sidd = PairedCSVDataset(
        csv_path=r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
        transform=transform
    )
    print(f"SIDD          : {len(sidd)} samples")
    datasets.append(sidd)

    csd = PairedFolderDataset(
        input_dir=r"E:/restormer+volterra/data/CSD/Train/Snow",
        target_dir=r"E:/restormer+volterra/data/CSD/Train/Gt",
        transform=transform
    )
    print(f"CSD           : {len(csd)} samples")
    datasets.append(csd)

    bsds = PairedFolderDataset(
        input_dir=r"E:/restormer+volterra/data/BSDS500/images/train",
        target_dir=r"E:/restormer+volterra/data/BSDS500/ground_truth/train",
        transform=transform
    )
    print(f"BSDS500 JPEG  : {len(bsds)} samples")
    datasets.append(bsds)

    total = sum(len(d) for d in datasets)
    print(f"📊 Total training samples: {total} images\n")

    return ConcatDataset(datasets)

# ---------------------- Training Loop ----------------------
def train_all_tasks():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print("\n🚀 Starting Unified Multi-Task Training from scratch\n")
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        transform = get_transform(epoch)
        dataset = get_all_datasets(transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        total_loss = total_psnr = total_ssim = count = 0
        loop = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{TOTAL_EPOCHS}]", leave=False)

        for inputs, targets in loop:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            outputs = outputs.detach().clamp(0, 1).cpu().numpy()
            targets = targets.cpu().numpy()

            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            loop.set_postfix(loss=loss.item())

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"📣 Epoch {epoch+1:03d}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

if __name__ == "__main__":
    train_all_tasks() """


# new code
""" 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra

# ------------------ Dataset 정의 ------------------
class PairedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])

        self.target_paths = []
        for inp_path in self.input_paths:
            fname = os.path.basename(inp_path)
            # 가능한 확장자 우선 순위로 매칭
            found = False
            for ext in valid_ext:
                gt_path = os.path.join(target_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(gt_path):
                    self.target_paths.append(gt_path)
                    found = True
                    break
            if not found:
                print(f"[⚠️] GT not found for {fname}")
                self.input_paths.remove(inp_path)

        assert len(self.input_paths) == len(self.target_paths), "Input/target mismatch"
        self.transform = transform


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)

class PairedCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)

# ------------------ 설정 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = r"E:/restormer+volterra/checkpoints/#01_all_tasks_new code"
os.makedirs(SAVE_DIR, exist_ok=True)
EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-4

# ------------------ Transform ------------------
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

# ------------------ Dataset 로딩 ------------------
def get_all_datasets(transform):
    print("\n📦 Preparing datasets...")

    rain100h = PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100H/train/rain",
        r"E:/restormer+volterra/data/rain100H/train/norain",
        transform
    )
    print(f"Rain100H      : {len(rain100h)} samples")

    rain100l = PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100L/train/rain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        transform
    )
    print(f"Rain100L      : {len(rain100l)} samples")

    gopro = PairedCSVDataset(
        r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
        transform
    )
    print(f"GoPro         : {len(gopro)} samples")

    sidd = PairedCSVDataset(
        r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
        transform
    )
    print(f"SIDD          : {len(sidd)} samples")

    csd = PairedFolderDataset(
        r"E:/restormer+volterra/data/CSD/Train/Snow",
        r"E:/restormer+volterra/data/CSD/Train/Gt",
        transform
    )
    print(f"CSD           : {len(csd)} samples")

    bsds = PairedFolderDataset(
        r"E:/restormer+volterra/data/BSDS500/images/train",
        r"E:/restormer+volterra/data/BSDS500/ground_truth/train",
        transform
    )
    print(f"BSDS500 JPEG  : {len(bsds)} samples")

    datasets = [rain100h, rain100l, gopro, sidd, csd, bsds]
    total = sum(len(d) for d in datasets)
    print(f"📊 Total training samples: {total} images\n")

    return ConcatDataset(datasets)

# ------------------ 학습 루프 ------------------
def train_all_tasks():
    model = RestormerVolterra().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    print("\n🚀 Starting Unified Training...\n")

    for epoch in range(1, EPOCHS + 1):
        transform = get_transform(epoch)
        dataset = get_all_datasets(transform)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        total_loss = total_psnr = total_ssim = count = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Metric 계산
            outputs = outputs.detach().clamp(0, 1).cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            loop.set_postfix(loss=loss.item())

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"📣 Epoch {epoch:03d}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

# ------------------ 실행 ------------------
if __name__ == "__main__":
    train_all_tasks()


 """

# 데이터개수 맞추기
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra

# ------------------ Dataset 정의 ------------------
class PairedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        self.input_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ])

        self.target_paths = []
        for inp_path in self.input_paths:
            fname = os.path.basename(inp_path)
            found = False
            for ext in valid_ext:
                gt_path = os.path.join(target_dir, os.path.splitext(fname)[0] + ext)
                if os.path.exists(gt_path):
                    self.target_paths.append(gt_path)
                    found = True
                    break
            if not found:
                print(f"[⚠️] GT not found for {fname}")
                self.input_paths.remove(inp_path)

        assert len(self.input_paths) == len(self.target_paths), "Input/target mismatch"
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)

class PairedCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths = df.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)

# ------------------ 설정 ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = r"E:/restormer+volterra/checkpoints/#01_all_tasks_balanced"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-4
MAX_SAMPLES = 160  # 모든 데이터셋에서 추출할 개수

# ------------------ Transform ------------------
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

# ------------------ Dataset 샘플링 ------------------
def subset_dataset(dataset, max_samples):
    total = len(dataset)
    if total <= max_samples:
        return dataset
    indices = random.sample(range(total), max_samples)
    return Subset(dataset, indices)

# ------------------ Dataset 로딩 ------------------
def get_all_datasets(transform):
    print("\n📦 Preparing balanced datasets...")

    datasets = []

    rain100h = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100H/train/rain",
        r"E:/restormer+volterra/data/rain100H/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100H      : {len(rain100h)} samples")
    datasets.append(rain100h)

    rain100l = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/rain100L/train/rain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        transform
    ), MAX_SAMPLES)
    print(f"Rain100L      : {len(rain100l)} samples")
    datasets.append(rain100l)

    gopro = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"GoPro         : {len(gopro)} samples")
    datasets.append(gopro)

    sidd = subset_dataset(PairedCSVDataset(
        r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
        transform
    ), MAX_SAMPLES)
    print(f"SIDD          : {len(sidd)} samples")
    datasets.append(sidd)

    csd = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/CSD/Train/Snow",
        r"E:/restormer+volterra/data/CSD/Train/Gt",
        transform
    ), MAX_SAMPLES)
    print(f"CSD           : {len(csd)} samples")
    datasets.append(csd)

    bsds = subset_dataset(PairedFolderDataset(
        r"E:/restormer+volterra/data/BSDS500/images/train",
        r"E:/restormer+volterra/data/BSDS500/ground_truth/train",
        transform
    ), MAX_SAMPLES)
    print(f"BSDS500 JPEG  : {len(bsds)} samples")
    datasets.append(bsds)

    total = sum(len(d) for d in datasets)
    print(f"📊 Total training samples (balanced): {total} images\n")

    return ConcatDataset(datasets)

# ------------------ 학습 루프 ------------------
def train_all_tasks():
    model = RestormerVolterra().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    print("\n🚀 Starting Balanced Multi-Task Training...\n")

    for epoch in range(1, EPOCHS + 1):
        transform = get_transform(epoch)
        dataset = get_all_datasets(transform)
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        model.train()
        total_loss = total_psnr = total_ssim = count = 0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Metric 계산
            outputs = outputs.detach().clamp(0, 1).cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            loop.set_postfix(loss=loss.item())

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"📣 Epoch {epoch:03d}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

# ------------------ 실행 ------------------
if __name__ == "__main__":
    train_all_tasks()
