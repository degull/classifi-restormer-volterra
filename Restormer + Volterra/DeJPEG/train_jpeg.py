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
import sys

from models.restormer_volterra import RestormerVolterra

# ----------------------- Config -----------------------
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR    = r"E:/restormer+volterra/data/BSDS500/images/train"
GT_DIR     = r"E:/restormer+volterra/data/BSDS500/ground_truth/train"
SAVE_DIR   = r"E:/restormer+volterra/checkpoints/jpeg_bsds500"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------- Dataset -----------------------
class JPEGDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.gt_paths  = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".png") or f.endswith(".bmp")])

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        input_tensor = self.transform(img)

        gt_path = self.gt_paths[idx].replace(".jpg", ".png")
        gt_img = Image.open(gt_path).convert('RGB')
        gt_tensor = self.transform(gt_img)

        return input_tensor, gt_tensor


# ----------------------- Training Function -----------------------
def main():
    dataset = JPEGDataset(IMG_DIR, GT_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # DEBUG: num_workers=0

    model = RestormerVolterra().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    psnr_all = []
    ssim_all = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_psnr, total_ssim = 0.0, 0.0
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)
        for inputs, targets in loop:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                outputs = model(inputs)
                loss = nn.functional.l1_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            outputs = outputs.clamp(0, 1).detach().cpu().numpy()
            targets = targets.cpu().numpy()

            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)

        avg_psnr = total_psnr / len(dataset)
        avg_ssim = total_ssim / len(dataset)
        psnr_all.append(avg_psnr)
        ssim_all.append(avg_ssim)

        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time) / 1000.0  # seconds

        print(f"[Epoch {epoch:03d}] PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}  Time: {elapsed:.1f}s")

        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

    print("\n==== Final Summary ====")
    for i, (p, s) in enumerate(zip(psnr_all, ssim_all), 1):
        print(f"Epoch {i:03d}: PSNR={p:.2f}, SSIM={s:.4f}")

# ----------------------- Entry -----------------------
if __name__ == '__main__':
    main()
