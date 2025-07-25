# train_and_test_motion_blur.py
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
from models.restormer_volterra import RestormerVolterra
from re_dataset.gopro_dataset import GoProDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_gopro_joint"
os.makedirs(SAVE_DIR, exist_ok=True)

GOPRO_TRAIN_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
GOPRO_TEST_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv"
HIDE_TEST_DIR = r"E:/restormer+volterra/data/HIDE"

resize_schedule = {0: 128, 30: 192, 60: 256}


def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def evaluate_folder(model, input_dir, target_dir, transform, name):
    model.eval()

    # 하위 디렉토리 포함 모든 이미지 탐색
    input_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                input_files.append(os.path.join(root, f))

    if not input_files:
        print(f"[WARN] {name} 평가에 사용될 파일이 없습니다.")
        return 0, 0

    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for input_path in sorted(input_files):
            fname = os.path.basename(input_path)
            target_path = os.path.join(target_dir, fname)

            if not os.path.isfile(target_path):
                continue

            input_img = transform(Image.open(input_path).convert("RGB"))
            target_img = transform(Image.open(target_path).convert("RGB"))

            input_img = input_img.unsqueeze(0).to(DEVICE)
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
    print(f"✅ {name}  PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"\n[INFO] Training GoPro, Evaluation on GoPro / HIDE\n")

    history = {
        'train_psnr': [], 'train_ssim': [],
        'gopro_psnr': [], 'gopro_ssim': [],
        'hide_psnr': [], 'hide_ssim': [],
    }

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)
        train_ds = GoProDataset(csv_path=GOPRO_TRAIN_CSV, transform=transform)
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
            ssim = compute_ssim(ref_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            tot_psnr += psnr
            tot_ssim += ssim
            count += 1

            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", ssim=f"{ssim:.3f}")

        avg_psnr = tot_psnr / count
        avg_ssim = tot_ssim / count
        avg_loss = epoch_loss / len(train_dl)

        print(f"[Epoch {epoch+1:3d}] Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} | Train SSIM: {avg_ssim:.4f}")

        # ───── 테스트 평가 ─────
        gopro_psnr, gopro_ssim = evaluate_folder(
            model,
            input_dir=os.path.join(os.path.dirname(GOPRO_TEST_CSV), "test/GOPR0384_11_00/blur"),
            target_dir=os.path.join(os.path.dirname(GOPRO_TEST_CSV), "test/GOPR0384_11_00/sharp"),
            transform=transform,
            name="GoPro (CSV)"
        )
        hide_psnr, hide_ssim = evaluate_folder(
            model,
            input_dir=os.path.join(HIDE_TEST_DIR, "test"),
            target_dir=os.path.join(HIDE_TEST_DIR, "GT"),
            transform=transform,
            name="HIDE"
        )

        history['train_psnr'].append(avg_psnr)
        history['train_ssim'].append(avg_ssim)
        history['gopro_psnr'].append(gopro_psnr)
        history['gopro_ssim'].append(gopro_ssim)
        history['hide_psnr'].append(hide_psnr)
        history['hide_ssim'].append(hide_ssim)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    # ───── 전체 결과 출력 ─────
    print("\n📊 [전체 Epoch별 요약 PSNR/SSIM]\n")
    header = f"{'Ep':>3} | {'Train':>12} | {'GoPro':>12} | {'HIDE':>12}"
    print(header)
    print("-" * len(header))
    for i in range(EPOCHS):
        print(f"{i+1:3d} | "
              f"{history['train_psnr'][i]:.2f}/{history['train_ssim'][i]:.3f} | "
              f"{history['gopro_psnr'][i]:.2f}/{history['gopro_ssim'][i]:.3f} | "
              f"{history['hide_psnr'][i]:.2f}/{history['hide_ssim'][i]:.3f}")


if __name__ == "__main__":
    main()



# ✅ GoPro (CSV)  PSNR: 36.99 | SSIM: 0.9871
# ✅ HIDE  PSNR: 33.70 | SSIM: 0.9601

# ✅ GoPro (CSV)  PSNR: 37.20 | SSIM: 0.9867
# ✅ HIDE  PSNR: 33.99 | SSIM: 0.9602


