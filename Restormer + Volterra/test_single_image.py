# python ver
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
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"


# ✅ 단일 이미지 경로 (rain100L)
DISTORTED_PATH = r"E:\restormer+volterra\data\rain100H\test\rain\norain-22.png"
REFERENCE_PATH = r"E:\restormer+volterra\data\rain100H\test\norain\norain-22.png"
SAVE_PATH = r"E:\restormer+volterra\results\comparison_output_rain100H_22.png"



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


# matlab ver
""" 
import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import autocast

# ─────────────────── 설정 ───────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"

DISTORTED = r"E:\restormer+volterra\data\rain100H\test\rain\norain-22.png"
REFERENCE  = r"E:\restormer+volterra\data\rain100H\test\norain\norain-22.png"
SAVE_PATH  = r"E:\restormer+volterra\results\comparison_output_rain100H_22.png"

# ─────────────────── 유틸 ───────────────────
to_tensor = T.ToTensor()

def pad_to_multiple(x, factor=8):
    b, c, h, w = x.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    pad_h, pad_w = H - h, W - w
    padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, h, w  # 원본 크기 반환

def rgb2y(img_rgb):          # RGB float32 0-1 → Y float32 0-255
    return (0.257 * img_rgb[..., 0] +
            0.504 * img_rgb[..., 1] +
            0.098 * img_rgb[..., 2] + 16/255) * 255

def add_label(img, txt):
    canvas = Image.new("RGB", (img.width, img.height + 24), (255, 255, 255))
    canvas.paste(img, (0, 24))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    tw = d.textlength(txt, font=font)
    d.text(((img.width - tw) // 2, 4), txt, fill=(0, 0, 0), font=font)
    return canvas

# ─────────────────── 데이터 로드 ───────────────────
dist_pil = Image.open(DISTORTED).convert("RGB")
ref_pil  = Image.open(REFERENCE).convert("RGB")

dist_t = to_tensor(dist_pil).unsqueeze(0).to(DEVICE)
ref_t  = to_tensor(ref_pil).unsqueeze(0).to(DEVICE)

# ─────────────────── 모델 ───────────────────
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ─────────────────── 추론 ───────────────────
dist_pad, orig_h, orig_w = pad_to_multiple(dist_t, factor=8)

with torch.no_grad():
    with autocast():
        restored_pad = model(dist_pad)

restored_t = restored_pad[:, :, :orig_h, :orig_w]  # crop to original size

# ─────────────────── NumPy & Y 채널 ───────────────────
ref_np      = ref_t.squeeze(0).cpu().numpy().transpose(1,2,0)
dist_np     = dist_t.squeeze(0).cpu().numpy().transpose(1,2,0)
restored_np = restored_t.squeeze(0).cpu().numpy().transpose(1,2,0)

ref_y   = rgb2y(ref_np)
dist_y  = rgb2y(dist_np)
rest_y  = rgb2y(restored_np)

# ─────────────────── PSNR / SSIM ───────────────────
psnr_in  = peak_signal_noise_ratio(ref_y, dist_y, data_range=255)
ssim_in  = structural_similarity(ref_y, dist_y, data_range=255)
psnr_out = peak_signal_noise_ratio(ref_y, rest_y, data_range=255)
ssim_out = structural_similarity(ref_y, rest_y, data_range=255)

print("\n📌 [Y 채널 기준 PSNR / SSIM]")
print(f"📎 입력  → GT : PSNR {psnr_in:.2f}  SSIM {ssim_in:.4f}")
print(f"📎 복원결과 → GT: PSNR {psnr_out:.2f}  SSIM {ssim_out:.4f}")

# ─────────────────── 시각화 & 저장 ───────────────────
ref_img      = Image.fromarray((ref_np      * 255).astype(np.uint8))
dist_img     = Image.fromarray((dist_np     * 255).astype(np.uint8))
restored_img = Image.fromarray((np.clip(restored_np,0,1)*255).astype(np.uint8))

final = Image.new("RGB", (ref_img.width*3, ref_img.height+24), (255,255,255))
final.paste(add_label(ref_img,      "Reference"), (0,0))
final.paste(add_label(dist_img,     "Distorted"), (ref_img.width,0))
final.paste(add_label(restored_img, "Restored"),  (ref_img.width*2,0))
final.save(SAVE_PATH)

print(f"✅ 비교 이미지 저장 완료 → {SAVE_PATH}")
 """