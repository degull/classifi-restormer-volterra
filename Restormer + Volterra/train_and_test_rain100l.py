import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image

from restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset

# ───────────── 설정 ─────────────
BATCH_SIZE = 2
EPOCHS     = 100
LR         = 2e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR  = r"E:/restormer+volterra/data/rain100L/train"
TEST_DIR   = r"E:/restormer+volterra/data/rain100L/test"
SAVE_DIR   = r"checkpoints/restormer_volterra_rain100l_joint"
os.makedirs(SAVE_DIR, exist_ok=True)

resize_schedule = {0: 128, 30: 192, 60: 256}

def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

def evaluate(model, test_dir, transform):
    model.eval()
    input_dir = os.path.join(test_dir, "rain")
    target_dir = os.path.join(test_dir, "norain")

    input_files = sorted(os.listdir(input_dir))
    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, fname)
            if not os.path.isfile(target_path):
                continue

            input_img = transform(Image.open(input_path).convert("RGB"))
            target_img = transform(Image.open(target_path).convert("RGB"))

            input_img  = input_img.unsqueeze(0).to(DEVICE)
            target_img = target_img.unsqueeze(0).to(DEVICE)

            output = model(input_img)

            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            tgt_np = target_img[0].cpu().numpy().transpose(1, 2, 0)

            psnr = compute_psnr(tgt_np, out_np, data_range=1.0)
            ssim = compute_ssim(tgt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim

# ───────────── 학습 루프 ─────────────
def main():
    model     = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    print(f"\n[INFO] Training + Evaluation on Rain100L (Train+Test)\n")

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        train_ds = Rain100LDataset(root_dir=TRAIN_DIR, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1:3d}] Input size: {transform.transforms[0].size} | Train Samples: {len(train_ds)}")

        model.train()
        epoch_loss = tot_psnr = tot_ssim = 0.0
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

            ref_np = reference[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            psnr = compute_psnr(ref_np, out_np, data_range=1.0)
            ssim = compute_ssim(ref_np, out_np, data_range=1.0,
                                channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count
        avg_loss = epoch_loss / len(train_dl)

        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} | Train SSIM: {avg_ssim:.4f}")

        # ───── 테스트 평가 ─────
        test_psnr, test_ssim = evaluate(model, TEST_DIR, transform)
        print(f"✅ [Epoch {epoch+1:3d}] Test  PSNR: {test_psnr:.2f} | Test  SSIM: {test_ssim:.4f}")

        # ───── 체크포인트 저장 ─────
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()



# ✅ [Epoch  40] Test  PSNR: 33.02 | Test  SSIM: 0.9404
# ✅ [Epoch  41] Test  PSNR: 32.64 | Test  SSIM: 0.9413
# ✅ [Epoch  62] Test  PSNR: 33.36 | Test  SSIM: 0.9442
# ✅ [Epoch  73] Test  PSNR: 34.30 | Test  SSIM: 0.9563
# ✅ [Epoch  89] Test  PSNR: 34.31 | Test  SSIM: 0.9575
# ✅ [Epoch 100] Test  PSNR: 34.28 | Test  SSIM: 0.9564