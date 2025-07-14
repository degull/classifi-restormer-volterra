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
from re_dataset.gopro_dataset import GoProDataset

# ───── 경로 설정 ─────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
SAVE_DIR = r"E:/restormer+volterra/checkpoints/restormer_volterra_gopro_joint"
os.makedirs(SAVE_DIR, exist_ok=True)

GOPRO_TRAIN_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv"
GOPRO_TEST_CSV = r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv"
HIDE_TEST_DIR = r"E:/restormer+volterra/data/HIDE"
REALBLUR_R_DIR = r"E:/restormer+volterra/data/RealBlur_R"
REALBLUR_J_DIR = r"E:/restormer+volterra/data/RealBlur_J"

resize_schedule = {0: 128, 30: 192, 60: 256}

def get_transform(epoch: int):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

def evaluate_folder(model, input_dir, target_dir, transform, name):
    model.eval()
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
    if not input_files:
        print(f"[WARN] {name} 평가에 사용될 파일이 없습니다.")
        return 0, 0
    total_psnr = total_ssim = count = 0

    with torch.no_grad():
        for fname in input_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, os.path.basename(fname))
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


# ───── 학습 루프 ─────
def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    print(f"\n[INFO] Training GoPro, Evaluation on GoPro / HIDE / RealBlur-R / RealBlur-J\n")

    # 📝 Epoch별 결과 저장
    history = {
        'train_psnr': [], 'train_ssim': [],
        'gopro_psnr': [], 'gopro_ssim': [],
        'hide1_psnr': [], 'hide1_ssim': [],
        'hide2_psnr': [], 'hide2_ssim': [],
        'realr_psnr': [], 'realr_ssim': [],
        'realj_psnr': [], 'realj_ssim': [],
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
        hide_psnr1, hide_ssim1 = evaluate_folder(
            model,
            input_dir=os.path.join(HIDE_TEST_DIR, "test/test-close-ups"),
            target_dir=os.path.join(HIDE_TEST_DIR, "GT"),
            transform=transform,
            name="HIDE-close-ups"
        )
        hide_psnr2, hide_ssim2 = evaluate_folder(
            model,
            input_dir=os.path.join(HIDE_TEST_DIR, "test/test-long-shot"),
            target_dir=os.path.join(HIDE_TEST_DIR, "GT"),
            transform=transform,
            name="HIDE-long-shot"
        )
        realblur_r_psnr, realblur_r_ssim = evaluate_folder(
            model,
            input_dir=os.path.join(REALBLUR_R_DIR, "blur"),
            target_dir=os.path.join(REALBLUR_R_DIR, "sharp"),
            transform=transform,
            name="RealBlur-R"
        )
        realblur_j_psnr, realblur_j_ssim = evaluate_folder(
            model,
            input_dir=os.path.join(REALBLUR_J_DIR, "blur"),
            target_dir=os.path.join(REALBLUR_J_DIR, "sharp"),
            transform=transform,
            name="RealBlur-J"
        )

        # 📝 기록 저장
        history['train_psnr'].append(avg_psnr)
        history['train_ssim'].append(avg_ssim)
        history['gopro_psnr'].append(gopro_psnr)
        history['gopro_ssim'].append(gopro_ssim)
        history['hide1_psnr'].append(hide_psnr1)
        history['hide1_ssim'].append(hide_ssim1)
        history['hide2_psnr'].append(hide_psnr2)
        history['hide2_ssim'].append(hide_ssim2)
        history['realr_psnr'].append(realblur_r_psnr)
        history['realr_ssim'].append(realblur_r_ssim)
        history['realj_psnr'].append(realblur_j_psnr)
        history['realj_ssim'].append(realblur_j_ssim)

        # ───── 체크포인트 저장 ─────
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    # ───── 전체 결과 출력 ─────
    print("\n📊 [전체 Epoch별 요약 PSNR/SSIM]\n")
    header = f"{'Ep':>3} | {'Train':>12} | {'GoPro':>12} | {'HIDE-Close':>12} | {'HIDE-Long':>12} | {'RealR':>12} | {'RealJ':>12}"
    print(header)
    print("-" * len(header))
    for i in range(EPOCHS):
        print(f"{i+1:3d} | "
              f"{history['train_psnr'][i]:.2f}/{history['train_ssim'][i]:.3f} | "
              f"{history['gopro_psnr'][i]:.2f}/{history['gopro_ssim'][i]:.3f} | "
              f"{history['hide1_psnr'][i]:.2f}/{history['hide1_ssim'][i]:.3f} | "
              f"{history['hide2_psnr'][i]:.2f}/{history['hide2_ssim'][i]:.3f} | "
              f"{history['realr_psnr'][i]:.2f}/{history['realr_ssim'][i]:.3f} | "
              f"{history['realj_psnr'][i]:.2f}/{history['realj_ssim'][i]:.3f}")

if __name__ == "__main__":
    main()