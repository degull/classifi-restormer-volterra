# test_single_image.py
# Reference (GT) 이미지 | Distorted (왜곡된 입력) 이미지 | Restored (복원 결과) 이미지

import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.cuda.amp import autocast

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_97.pth"

# ✅ 단일 이미지 경로 (KADID)
DISTORTED_PATH = r"E:\restormer+volterra\data\tid2013\distorted_images\i01_19_5.bmp"
REFERENCE_PATH = r"E:\restormer+volterra\data\tid2013\reference_images\I01.BMP"
SAVE_PATH = r"E:\restormer+volterra\results\tt.png"


""" # ✅ 단일 이미지 경로 (HIDE)
DISTORTED_PATH = r"E:\restormer+volterra\data\HIDE\test\test-close-ups\23fromGOPR0977.png"
REFERENCE_PATH = r"E:\restormer+volterra\data\HIDE\GT\1fromGOPR0977.png"
SAVE_PATH = r"E:\restormer+volterra\results\comparison_output_hide.png" """


""" # ✅ 단일 이미지 경로 (rain100L)
DISTORTED_PATH = r"E:\restormer+volterra\data\rain100L\test\rain\norain-1.png"
REFERENCE_PATH = r"E:\restormer+volterra\data\rain100L\test\norain\norain-1.png"
SAVE_PATH = r"E:\restormer+volterra\results\comparison_output_rain100l.png" """


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
        restored_img = model(distorted_img)

# ✅ NumPy 변환
distorted_np = distorted_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
reference_np = reference_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
restored_np = restored_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

distorted_np = np.clip(distorted_np, 0, 1)
reference_np = np.clip(reference_np, 0, 1)
restored_np = np.clip(restored_np, 0, 1)

# ✅ PSNR / SSIM 계산
psnr_restored = compute_psnr(reference_np, restored_np, data_range=1.0)
ssim_restored = compute_ssim(reference_np, restored_np, data_range=1.0, channel_axis=2)
psnr_distorted = compute_psnr(reference_np, distorted_np, data_range=1.0)
ssim_distorted = compute_ssim(reference_np, distorted_np, data_range=1.0, channel_axis=2)

print(f"📌 [참조 vs 왜곡] PSNR: {psnr_distorted:.2f} dB, SSIM: {ssim_distorted:.4f}")
print(f"✅ [참조 vs 복원] PSNR: {psnr_restored:.2f} dB, SSIM: {ssim_restored:.4f}")

# ✅ PIL 이미지 변환
ref_img = Image.fromarray((reference_np * 255).astype(np.uint8))
dist_img = Image.fromarray((distorted_np * 255).astype(np.uint8))
restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))

# ✅ 텍스트 라벨 추가 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))  # 위에 공간 추가
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

ref_labeled = add_label(ref_img, "Reference")
dist_labeled = add_label(dist_img, "Distorted")
restored_labeled = add_label(restored_img, "Restored")

# ✅ 가로로 이어 붙이기
final_img = Image.new('RGB', (256 * 3, 280))
final_img.paste(ref_labeled, (0, 0))
final_img.paste(dist_labeled, (256, 0))
final_img.paste(restored_labeled, (512, 0))
final_img.save(SAVE_PATH)

print(f"✅ 라벨 포함 이미지 저장 완료: {SAVE_PATH}")
