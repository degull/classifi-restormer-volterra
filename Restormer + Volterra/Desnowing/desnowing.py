import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.amp import autocast, GradScaler
from restormer_volterra import RestormerVolterra


class CSDDataset(Dataset):
    def __init__(self, snow_dir, gt_dir, transform=None):
        self.snow_files = sorted(os.listdir(snow_dir))
        self.snow_dir = snow_dir
        self.gt_dir = gt_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.snow_files)

    def __getitem__(self, idx):
        snow_path = os.path.join(self.snow_dir, self.snow_files[idx])
        gt_path = os.path.join(self.gt_dir, self.snow_files[idx])
        snow = Image.open(snow_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        return self.transform(snow), self.transform(gt)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_csd"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_SNOW_DIR = r"E:/restormer+volterra/data/CSD/Train/Snow"
TRAIN_GT_DIR = r"E:/restormer+volterra/data/CSD/Train/Gt"
TEST_SNOW_DIR = r"E:/restormer+volterra/data/CSD/Test/Snow"
TEST_GT_DIR = r"E:/restormer+volterra/data/CSD/Test/Gt"

resize_schedule = {0: 128, 30: 192, 60: 256}


def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def evaluate_csd(model, dataloader):
    model.eval()
    total_psnr = total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for snow, gt in dataloader:
            snow = snow.to(DEVICE)
            gt = gt.to(DEVICE)
            output = model(snow)
            gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"✅ CSD Test PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"\n[INFO] Training on CSD Train / Testing on CSD Test\n")

    history = {
        'train_psnr': [], 'train_ssim': [],
        'test_psnr': [], 'test_ssim': [],
    }

    test_ds = CSDDataset(TEST_SNOW_DIR, TEST_GT_DIR, transform=get_transform(0))
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = CSDDataset(TRAIN_SNOW_DIR, TRAIN_GT_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)} | Test Samples: {len(test_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
        count = 0

        loop = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False)

        for snow, gt in loop:
            snow = snow.to(DEVICE)
            gt = gt.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                output = model(snow)
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

        test_psnr, test_ssim = evaluate_csd(model, test_dl)

        history['train_psnr'].append(avg_psnr)
        history['train_ssim'].append(avg_ssim)
        history['test_psnr'].append(test_psnr)
        history['test_ssim'].append(test_ssim)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    print("\n📊 [전체 Epoch별 요약 PSNR/SSIM]\n")
    header = f"{'Ep':>3} | {'Train':>12} | {'Test':>12}"
    print(header)
    print("-" * len(header))
    for i in range(EPOCHS):
        print(f"{i+1:3d} | "
              f"{history['train_psnr'][i]:.2f}/{history['train_ssim'][i]:.3f} | "
              f"{history['test_psnr'][i]:.2f}/{history['test_ssim'][i]:.3f}")


if __name__ == "__main__":
    main()


"""
[Epoch  21] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  21] Train Loss: 0.000081 | Train PSNR: 41.19 | Train SSIM: 0.9910
✅ CSD Test PSNR: 39.38 | SSIM: 0.9897
[Epoch  22] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  22] Train Loss: 0.000076 | Train PSNR: 41.52 | Train SSIM: 0.9915
✅ CSD Test PSNR: 40.72 | SSIM: 0.9906
[Epoch  23] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  23] Train Loss: 0.000072 | Train PSNR: 41.76 | Train SSIM: 0.9920
✅ CSD Test PSNR: 40.82 | SSIM: 0.9912
[Epoch  24] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  24] Train Loss: 0.000068 | Train PSNR: 41.98 | Train SSIM: 0.9924
✅ CSD Test PSNR: 41.17 | SSIM: 0.9915
[Epoch  25] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  25] Train Loss: 0.000064 | Train PSNR: 42.27 | Train SSIM: 0.9928
✅ CSD Test PSNR: 41.24 | SSIM: 0.9916
[Epoch  26] Input size: (128, 128) | Train Samples: 8000 | Test Samples: 2000
[Epoch  26] Train Loss: 0.000062 | Train PSNR: 42.35 | Train SSIM: 0.9930
✅ CSD Test PSNR: 41.05 | SSIM: 0.9918
✅ CSD Test PSNR: 41.27 | SSIM: 0.9922
"""