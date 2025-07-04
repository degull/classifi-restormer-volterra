# E:/MRVNet2D/Restormer + Volterra/test.py

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from restormer_volterra import RestormerVolterra
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.hide_dataset import HIDEDataset

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_98.pth"

# ✅ 데이터 경로
RAIN100L_DIR = 'E:/restormer+volterra/data/rain100L/test'
HIDE_DIR = 'E:/restormer+volterra/data/HIDE'

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 데이터셋 로딩
rain100l_dataset = Rain100LDataset(root_dir=RAIN100L_DIR, transform=transform)
hide_dataset = HIDEDataset(root_dir=HIDE_DIR, transform=transform)
test_dataset = ConcatDataset([rain100l_dataset, hide_dataset])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ 평가
psnr_total = 0.0
ssim_total = 0.0
num_images = len(test_loader)

with torch.no_grad():
    for distorted_img, reference_img in tqdm(test_loader, desc="Evaluating"):
        distorted_img = distorted_img.to(DEVICE)
        reference_img = reference_img.to(DEVICE)

        with autocast():
            output = model(distorted_img)

        # Tensor → Numpy
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reference_np = reference_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        output_np = np.clip(output_np, 0, 1)
        reference_np = np.clip(reference_np, 0, 1)

        # PSNR / SSIM 계산
        psnr = compute_psnr(reference_np, output_np, data_range=1.0)
        ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2)

        psnr_total += psnr
        ssim_total += ssim

# ✅ 평균 결과 출력
print(f"\n✅ Average PSNR: {psnr_total / num_images:.2f} dB")
print(f"✅ Average SSIM: {ssim_total / num_images:.4f}")


"""
97
✅ Average PSNR: 28.76 dB
✅ Average SSIM: 0.8687
"""


