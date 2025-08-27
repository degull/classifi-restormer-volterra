# test_complex_distortion.py
# 이미지 저장까지
import os, sys, glob
sys.path.append(r"E:/restormer+volterra/Restormer + Volterra/")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra
from pipeline import apply_random_distortions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_100_ssim0.9177_psnr32.58.pth"

# ----------------- 유틸 -----------------
def load_img(path): 
    return Image.open(path).convert("RGB")

def tensor_to_numpy(t):
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1,2,0))
    return np.clip(arr,0,1)

def save_img(path, arr01):
    arr = (np.clip(arr01,0,1)*255.0+0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)

def load_model(ckpt_path):
    model = RestormerVolterra().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt if "state_dict" not in ckpt else ckpt["state_dict"])
    model.eval()
    return model

# ----------------- 실행 -----------------
if __name__ == "__main__":
    model = load_model(CKPT)
    tf = transforms.ToTensor()

    # ★ 모든 GT 경로 모으기 ★
    gt_dirs = [
        r"E:/restormer+volterra/data/CSD/Train/Gt",
        r"E:/restormer+volterra/data/CSD/Test/Gt",
        r"E:/restormer+volterra/data/HIDE/GT",
        r"E:/restormer+volterra/data/rain100H/train/norain",
        r"E:/restormer+volterra/data/rain100H/test/norain",
        r"E:/restormer+volterra/data/rain100L/train/norain",
        r"E:/restormer+volterra/data/rain100L/test/norain",
        r"E:/restormer+volterra/data/SIDD/Data"  # SIDD는 안에서 GT_SRGB만 뽑음
    ]

    gt_files = []
    for d in gt_dirs:
        if "SIDD" in d:
            gt_files += glob.glob(os.path.join(d, "**", "GT_SRGB_*.PNG"), recursive=True)
        else:
            gt_files += glob.glob(os.path.join(d, "*.*"))

    save_dir = r"E:/restormer+volterra/results/multiple_distortion"
    os.makedirs(save_dir, exist_ok=True)

    total_psnr, total_ssim = 0, 0
    count = 0

    for img_path in gt_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        clean = load_img(img_path)

        # ARNIQA-style 복합 왜곡 적용
        distorted, applied_info = apply_random_distortions(clean, Ndist=3, return_info=True)

        input_tensor = tf(distorted).unsqueeze(0).to(DEVICE)
        gt_tensor    = tf(clean).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            restored = model(input_tensor)

        restored_np = tensor_to_numpy(restored[0])
        gt_np       = tensor_to_numpy(gt_tensor[0])
        inp_np      = tensor_to_numpy(input_tensor[0])

        psnr = compute_psnr(gt_np, restored_np, data_range=1.0)
        ssim = compute_ssim(gt_np, restored_np, channel_axis=-1, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # 터미널 출력
        print(f"[{count}] {os.path.basename(img_path)} | Distortions: {applied_info} | PSNR={psnr:.2f}, SSIM={ssim:.4f}")

        # 이미지 저장
        save_img(os.path.join(save_dir, f"{fname}_input.png"), inp_np)
        save_img(os.path.join(save_dir, f"{fname}_restored.png"), restored_np)
        save_img(os.path.join(save_dir, f"{fname}_gt.png"), gt_np)

    # 전체 평균 출력
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print("="*60)
        print(f"Processed {count} images")
        print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
