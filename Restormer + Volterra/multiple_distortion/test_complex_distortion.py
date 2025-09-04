# test_complex_distortion.py
import os, sys, glob, time
sys.path.append(r"E:/restormer+volterra/Restormer + Volterra/")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm   # ✅ 진행바

from models.restormer_volterra import RestormerVolterra
from pipeline import apply_random_distortions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced_160\epoch_100_ssim0.9177_psnr32.58.pth"

# ----------------- 유틸 -----------------
def load_img(path, resize=256): 
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return img

def tensor_to_numpy(t):
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1,2,0))
    return np.clip(arr,0,1)

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

    # GT pool
    gt_dirs = [
        r"E:/restormer+volterra/data/CSD/Test/Gt",
        r"E:/restormer+volterra/data/HIDE/GT",
        r"E:/restormer+volterra/data/rain100H/test/norain",
        r"E:/restormer+volterra/data/rain100L/test/norain",
        r"E:/restormer+volterra/data/SIDD/Data"
    ]

    gt_files = []
    for d in gt_dirs:
        if "SIDD" in d:
            gt_files += glob.glob(os.path.join(d, "**", "GT_SRGB_*.PNG"), recursive=True)
        else:
            gt_files += glob.glob(os.path.join(d, "*.*"))

    total_psnr, total_ssim = 0, 0
    count = 0
    start_time = time.time()

    # tqdm 진행률 표시
    for img_path in tqdm(gt_files, desc="Processing images", ncols=100):
        clean = load_img(img_path, resize=256)
        distorted, info = apply_random_distortions(clean, Ndist=4, return_info=True)

        input_tensor = tf(distorted).unsqueeze(0).to(DEVICE)
        gt_tensor    = tf(clean).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            restored = model(input_tensor)

        psnr = compute_psnr(tensor_to_numpy(gt_tensor[0]), tensor_to_numpy(restored[0]), data_range=1.0)
        ssim = compute_ssim(tensor_to_numpy(gt_tensor[0]), tensor_to_numpy(restored[0]), channel_axis=-1, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # 개별 로그 (선택 사항)
        elapsed = time.time() - start_time
        avg_time = elapsed / count
        eta = avg_time * (len(gt_files) - count)
        print(f"[{count}/{len(gt_files)}] {os.path.basename(img_path)} "
              f"| Distortions: {info} | PSNR={psnr:.2f}, SSIM={ssim:.4f} "
              f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    # 최종 평균
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print("="*60)
        print(f"Processed {count} images")
        print(f"Final Average PSNR: {avg_psnr:.2f}")
        print(f"Final Average SSIM: {avg_ssim:.4f}")
        print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
