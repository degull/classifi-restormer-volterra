# test_rain100h.py
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
import cv2

# ───────────── 경로 설정 ─────────────
CHECKPOINT_PATH = r"E:/restormer+volterra/checkpoints/restormer_volterra_rain100h/epoch_100.pth"
RAIN100H_TEST_RAIN = r"E:/restormer+volterra/data/rain100H/test/rain"
RAIN100H_TEST_GT   = r"E:/restormer+volterra/data/rain100H/test/norain"
RESULT_SAVE_DIR    = r"E:/restormer+volterra/results/rain100H"

os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# ───────────── 기본 설정 ─────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

# ───────────── 모델 로드 ─────────────
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print(f"✅ 모델 로드 완료: {CHECKPOINT_PATH}")

# ───────────── 테스트 이미지 목록 ─────────────
img_list = sorted(os.listdir(RAIN100H_TEST_RAIN))

total_psnr, total_ssim = 0.0, 0.0

for filename in tqdm(img_list, desc="🔍 Testing"):
    rain_path = os.path.join(RAIN100H_TEST_RAIN, filename)
    gt_path   = os.path.join(RAIN100H_TEST_GT, filename)

    if not os.path.exists(gt_path):
        print(f"❌ GT 이미지 없음: {filename}")
        continue

    # ✅ 이미지 로드 (원본 해상도 유지)
    img_rain = Image.open(rain_path).convert("RGB")
    img_gt   = Image.open(gt_path).convert("RGB")

    img_rain_tensor = to_tensor(img_rain).unsqueeze(0).to(DEVICE)
    img_gt_tensor   = to_tensor(img_gt).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with autocast():
            restored_tensor = model(img_rain_tensor)

    # ✅ NumPy 변환
    restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    gt_np       = img_gt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_np = np.clip(restored_np, 0, 1)

    # ✅ PSNR/SSIM (Y 채널 기준)
    def rgb2y(img):
        return 0.257 * img[..., 0] * 255 + 0.504 * img[..., 1] * 255 + 0.098 * img[..., 2] * 255 + 16

    y_restored = rgb2y(restored_np)
    y_gt       = rgb2y(gt_np)

    if y_restored.shape != y_gt.shape:
        y_restored = cv2.resize(y_restored, (y_gt.shape[1], y_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    psnr = compute_psnr(y_gt, y_restored, data_range=255)
    ssim = compute_ssim(y_gt, y_restored, data_range=255)

    total_psnr += psnr
    total_ssim += ssim

    # ✅ 결과 저장
    save_path = os.path.join(RESULT_SAVE_DIR, filename)
    Image.fromarray((restored_np * 255).astype(np.uint8)).save(save_path)

# ───────────── 평균 결과 출력 ─────────────
avg_psnr = total_psnr / len(img_list)
avg_ssim = total_ssim / len(img_list)

print(f"\n📊 Rain100H 테스트셋 결과")
print(f"→ PSNR: {avg_psnr:.2f} dB")
print(f"→ SSIM: {avg_ssim:.4f}")
print(f"→ 복원 결과 저장 위치: {RESULT_SAVE_DIR}")
