# test_single_image.py
# ref o
# [Reference | Distorted | Restored]

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.cuda.amp import autocast

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\MRVNet2D\checkpoints\restormer_volterra_all\epoch_58.pth"

# ✅ 단일 이미지 경로
DISTORTED_PATH = r"E:\MRVNet2D\dataset\KADID10K\images\I81_24_05.png"
REFERENCE_PATH = r"E:\MRVNet2D\dataset\KADID10K\images\I81.png"
SAVE_PATH = r"E:\MRVNet2D\results\comparison_output.png"

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 이미지 로드 및 변환
distorted_img_pil = Image.open(DISTORTED_PATH).convert("RGB").resize((256, 256))
reference_img_pil = Image.open(REFERENCE_PATH).convert("RGB").resize((256, 256))

distorted_img = transform(distorted_img_pil).unsqueeze(0).to(DEVICE)
reference_img = transform(reference_img_pil).unsqueeze(0).to(DEVICE)

# ✅ 복원 수행
with torch.no_grad():
    with autocast():
        output = model(distorted_img)

# ✅ 복원 결과 → NumPy
output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
reference_np = reference_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
output_np = np.clip(output_np, 0, 1)
reference_np = np.clip(reference_np, 0, 1)

# ✅ PSNR / SSIM 계산
psnr = compute_psnr(reference_np, output_np, data_range=1.0)
ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2)
print(f"✅ PSNR: {psnr:.2f} dB")
print(f"✅ SSIM: {ssim:.4f}")

# ✅ 나란히 이미지 저장
distorted_np = distorted_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
distorted_np = np.clip(distorted_np, 0, 1)

ref_img = (reference_np * 255).astype(np.uint8)
dist_img = (distorted_np * 255).astype(np.uint8)
restored_img = (output_np * 255).astype(np.uint8)

# PIL로 변환 후 가로로 연결
ref_pil = Image.fromarray(ref_img)
dist_pil = Image.fromarray(dist_img)
restored_pil = Image.fromarray(restored_img)
concat_img = Image.new('RGB', (256 * 3, 256))
concat_img.paste(ref_pil, (0, 0))
concat_img.paste(dist_pil, (256, 0))
concat_img.paste(restored_pil, (512, 0))
concat_img.save(SAVE_PATH)

print(f"✅ Comparison image saved to: {SAVE_PATH}")


# ref x
""" 
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\MRVNet2D\checkpoints\restormer_volterra_all\epoch_58.pth"

# ✅ 입력 이미지 (왜곡 이미지)
DISTORTED_PATH = r"E:\MRVNet2D\dataset\KADID10K\images\I32_03_05.png"
SAVE_PATH = r"E:\MRVNet2D\results\restored_output_only.png"

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 이미지 로드 및 변환
distorted_img_pil = Image.open(DISTORTED_PATH).convert("RGB").resize((256, 256))
distorted_img = transform(distorted_img_pil).unsqueeze(0).to(DEVICE)

# ✅ 복원 수행
with torch.no_grad():
    with autocast():
        output = model(distorted_img)

# ✅ 결과 변환 및 저장
output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
output_np = np.clip(output_np, 0, 1)
restored_img = (output_np * 255).astype(np.uint8)
Image.fromarray(restored_img).save(SAVE_PATH)

print(f"✅ Restored image saved to: {SAVE_PATH}")
 """