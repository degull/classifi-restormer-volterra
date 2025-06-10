# 전체 데이터셋 복원
""" 
import os
import torch
from torchvision import transforms
from PIL import Image
from restormer_volterra import RestormerVolterra  # 동일한 모델 정의 파일
from kadid_dataset import KADID10KDataset         # 동일한 커스텀 데이터셋

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'E:/MRVNet2D/checkpoints/restormer_volterra_kadid/epoch_100.pth'
DATA_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
IMAGE_DIR = 'E:/MRVNet2D/dataset/KADID10K/images'
SAVE_DIR = 'E:/MRVNet2D/results/restored_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 데이터셋 준비
test_dataset = KADID10KDataset(csv_file=DATA_CSV, transform=transform)

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    for idx in range(len(test_dataset)):
        distorted, reference = test_dataset[idx]  # (tensor, tensor)
        input_img = distorted.unsqueeze(0).to(DEVICE)

        output = model(input_img)
        output_img = output.squeeze(0).cpu().clamp(0, 1)

        # 저장
        restored_pil = transforms.ToPILImage()(output_img)
        restored_pil.save(os.path.join(SAVE_DIR, f"restored_{idx:05d}.png"))

        if idx < 5:  # ✅ 처음 몇 개만 확인용 출력
            print(f"Saved: restored_{idx:05d}.png")

print("✅ 테스트 이미지 복원 완료.")
 """

# single image 복원
""" 
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'E:/MRVNet2D/checkpoints/restormer_volterra_kadid/epoch_100.pth'
DISTORTED_IMG_PATH = 'E:/MRVNet2D/tid2013/distorted_images/i14_07_5.bmp'    # 복원할 이미지
REFERENCE_IMG_PATH = 'E:/MRVNet2D/tid2013/reference_images/I14.BMP'    # 정답 이미지
SAVE_PATH = 'E:/MRVNet2D/results/comparison_result.png'

# ✅ 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# ✅ 이미지 로드
distorted = Image.open(DISTORTED_IMG_PATH).convert('RGB')
reference = Image.open(REFERENCE_IMG_PATH).convert('RGB')

distorted_tensor = transform(distorted).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
reference_tensor = transform(reference).unsqueeze(0).to(DEVICE)

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    output_tensor = model(distorted_tensor).clamp(0, 1)

# ✅ PSNR, SSIM 계산
output_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
reference_np = reference_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()


psnr = compute_psnr(reference_np, output_np, data_range=1.0)
ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2, win_size=7)

print(f"✅ PSNR: {psnr:.2f} dB")
print(f"✅ SSIM: {ssim:.4f}")

# ✅ 이미지 붙이기 및 저장
ref_img = to_pil(reference_tensor.squeeze(0).cpu())
dist_img = to_pil(distorted_tensor.squeeze(0).cpu())
out_img = to_pil(output_tensor.squeeze(0).cpu())

width, height = ref_img.size
concat_img = Image.new('RGB', (width * 3, height))
concat_img.paste(ref_img, (0, 0))
concat_img.paste(dist_img, (width, 0))
concat_img.paste(out_img, (width * 2, 0))
concat_img.save(SAVE_PATH)

print(f"✅ 비교 이미지 저장 완료: {SAVE_PATH}")
 """


# 복원 수치만 출력하는 테스트 코드
import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from torch.cuda.amp import autocast
from tqdm import tqdm

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\MRVNet2D\checkpoints\restormer_volterra_all\epoch_59.pth"
TEST_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
IMAGE_ROOT = 'E:/MRVNet2D/dataset/KADID10K/images'

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 초기화 및 weight 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ CSV 로딩
df = pd.read_csv(TEST_CSV)
psnr_total = 0.0
ssim_total = 0.0
num_images = len(df)

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=num_images, desc="Evaluating"):
        distorted_path = os.path.join(IMAGE_ROOT, row[0])
        reference_path = os.path.join(IMAGE_ROOT, row[1])

        # 이미지 로드 및 변환
        distorted_img = transform(Image.open(distorted_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        reference_img = transform(Image.open(reference_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        # 복원
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

# 58
#✅ Average PSNR: 28.35 dB
#✅ Average SSIM: 0.9059