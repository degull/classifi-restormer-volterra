# restore_kadid
""" import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"
DISTORTED_DIR = r"E:\restormer+volterra\data\KADID10K\images"
SAVE_DIR = r"E:\restormer+volterra\results\restored_kadid"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 라벨 추가 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))  # 위 공간 추가
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

# ✅ 폴더 순회
for filename in os.listdir(DISTORTED_DIR):
    # macOS 메타파일, 숨김파일, 확장자 검사
    if not (filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))) or filename.startswith("._"):
        continue
    if "_" not in filename:  # 참조 이미지일 경우 스킵
        continue

    distorted_path = os.path.join(DISTORTED_DIR, filename)
    ref_name = filename.split("_")[0] + ".png"
    reference_path = os.path.join(DISTORTED_DIR, ref_name)
    save_path = os.path.join(SAVE_DIR, filename.replace(".", "_restored."))

    # ✅ 이미지 로드
    distorted_pil = Image.open(distorted_path).convert("RGB").resize((256, 256))
    distorted_tensor = transform(distorted_pil).unsqueeze(0).to(DEVICE)

    reference_exists = os.path.exists(reference_path)
    if reference_exists:
        reference_pil = Image.open(reference_path).convert("RGB").resize((256, 256))
        reference_tensor = transform(reference_pil).unsqueeze(0).to(DEVICE)

    # ✅ 복원 수행
    with torch.no_grad():
        with autocast():
            restored_tensor = model(distorted_tensor)

    # ✅ 변환
    distorted_np = distorted_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    distorted_np = np.clip(distorted_np, 0, 1)
    restored_np = np.clip(restored_np, 0, 1)

    dist_img = Image.fromarray((distorted_np * 255).astype(np.uint8))
    restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))

    dist_labeled = add_label(dist_img, "Distorted")
    restored_labeled = add_label(restored_img, "Restored")

    # ✅ 라벨 포함 이미지 결합
    if reference_exists:
        reference_np = reference_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reference_np = np.clip(reference_np, 0, 1)
        ref_img = Image.fromarray((reference_np * 255).astype(np.uint8))
        ref_labeled = add_label(ref_img, "Reference")

        # PSNR/SSIM 출력
        psnr_dist = compute_psnr(reference_np, distorted_np, data_range=1.0)
        ssim_dist = compute_ssim(reference_np, distorted_np, data_range=1.0, channel_axis=2)
        psnr_rest = compute_psnr(reference_np, restored_np, data_range=1.0)
        ssim_rest = compute_ssim(reference_np, restored_np, data_range=1.0, channel_axis=2)

        print(f"📌 {filename} | PSNR (Dist): {psnr_dist:.2f}, SSIM (Dist): {ssim_dist:.3f} | "
              f"PSNR (Rest): {psnr_rest:.2f}, SSIM (Rest): {ssim_rest:.3f}")

        final = Image.new("RGB", (256 * 3, 280))
        final.paste(ref_labeled, (0, 0))
        final.paste(dist_labeled, (256, 0))
        final.paste(restored_labeled, (512, 0))
    else:
        print(f"⚠️ 참조 이미지 없음: {reference_path}")
        final = Image.new("RGB", (256 * 2, 280))
        final.paste(dist_labeled, (0, 0))
        final.paste(restored_labeled, (256, 0))

    final.save(save_path)

print(f"\n✅ 전체 복원 완료. 저장 경로: {SAVE_DIR}")
 """

# restore_tid2013.py
""" import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"
DISTORTED_DIR = r"E:\restormer+volterra\data\tid2013\distorted_images"
REFERENCE_DIR = r"E:\restormer+volterra\data\tid2013\reference_images"
SAVE_DIR = r"E:\restormer+volterra\results\restored_tid"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 라벨 추가 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))  # 위 공간 추가
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

# ✅ 복원 루프
for filename in os.listdir(DISTORTED_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or filename.startswith("._"):
        continue

    distorted_path = os.path.join(DISTORTED_DIR, filename)
    ref_id = filename.split("_")[0].lower()  # e.g., i01
    ref_name = ref_id.replace("i", "I") + ".BMP"  # e.g., I01.BMP
    reference_path = os.path.join(REFERENCE_DIR, ref_name)
    save_path = os.path.join(SAVE_DIR, filename.replace(".", "_restored."))

    try:
        distorted_pil = Image.open(distorted_path).convert("RGB").resize((256, 256))
        distorted_tensor = transform(distorted_pil).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"❌ 이미지 로딩 실패: {distorted_path} ({e})")
        continue

    reference_exists = os.path.exists(reference_path)
    if reference_exists:
        reference_pil = Image.open(reference_path).convert("RGB").resize((256, 256))
        reference_tensor = transform(reference_pil).unsqueeze(0).to(DEVICE)

    # ✅ 복원 수행
    with torch.no_grad():
        with autocast():
            restored_tensor = model(distorted_tensor)

    # ✅ 변환
    distorted_np = distorted_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    distorted_np = np.clip(distorted_np, 0, 1)
    restored_np = np.clip(restored_np, 0, 1)

    dist_img = Image.fromarray((distorted_np * 255).astype(np.uint8))
    restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))

    dist_labeled = add_label(dist_img, "Distorted")
    restored_labeled = add_label(restored_img, "Restored")

    # ✅ 라벨 포함 이미지 결합
    if reference_exists:
        reference_np = reference_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reference_np = np.clip(reference_np, 0, 1)
        ref_img = Image.fromarray((reference_np * 255).astype(np.uint8))
        ref_labeled = add_label(ref_img, "Reference")

        # PSNR/SSIM 출력
        psnr_dist = compute_psnr(reference_np, distorted_np, data_range=1.0)
        ssim_dist = compute_ssim(reference_np, distorted_np, data_range=1.0, channel_axis=2)
        psnr_rest = compute_psnr(reference_np, restored_np, data_range=1.0)
        ssim_rest = compute_ssim(reference_np, restored_np, data_range=1.0, channel_axis=2)

        print(f"📌 {filename} | PSNR (Dist): {psnr_dist:.2f}, SSIM (Dist): {ssim_dist:.3f} | "
              f"PSNR (Rest): {psnr_rest:.2f}, SSIM (Rest): {ssim_rest:.3f}")

        final = Image.new("RGB", (256 * 3, 280))
        final.paste(ref_labeled, (0, 0))
        final.paste(dist_labeled, (256, 0))
        final.paste(restored_labeled, (512, 0))
    else:
        print(f"⚠️ 참조 이미지 없음: {reference_path}")
        final = Image.new("RGB", (256 * 2, 280))
        final.paste(dist_labeled, (0, 0))
        final.paste(restored_labeled, (256, 0))

    final.save(save_path)

print(f"\n✅ 전체 복원 완료. 저장 경로: {SAVE_DIR}")
 """

# restore_csiq
""" import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"
DISTORTED_ROOT = r"E:\restormer+volterra\data\CSIQ\dst_imgs"
REFERENCE_DIR = r"E:\restormer+volterra\data\CSIQ\src_imgs"
SAVE_DIR = r"E:\restormer+volterra\results\restored_csiq"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 라벨 추가 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

# ✅ 전체 왜곡 폴더 순회
for distortion_type in os.listdir(DISTORTED_ROOT):
    subdir = os.path.join(DISTORTED_ROOT, distortion_type)
    if not os.path.isdir(subdir):
        continue

    print(f"📂 복원 시작: {distortion_type}")
    save_subdir = os.path.join(SAVE_DIR, distortion_type)
    os.makedirs(save_subdir, exist_ok=True)

    for filename in os.listdir(subdir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or filename.startswith("._"):
            continue

        distorted_path = os.path.join(subdir, filename)
        reference_name = filename.split('.')[0].split('_')[0] + ".png"
        reference_path = os.path.join(REFERENCE_DIR, reference_name)
        save_path = os.path.join(save_subdir, filename.replace(".", "_restored."))

        # ✅ 이미지 로드
        try:
            distorted_pil = Image.open(distorted_path).convert("RGB").resize((256, 256))
        except:
            print(f"⚠️ 이미지 열기 실패: {distorted_path}")
            continue

        distorted_tensor = transform(distorted_pil).unsqueeze(0).to(DEVICE)
        reference_exists = os.path.exists(reference_path)

        if reference_exists:
            reference_pil = Image.open(reference_path).convert("RGB").resize((256, 256))
            reference_tensor = transform(reference_pil).unsqueeze(0).to(DEVICE)

        # ✅ 복원 수행
        with torch.no_grad():
            with autocast():
                restored_tensor = model(distorted_tensor)

        # ✅ 변환
        distorted_np = distorted_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        distorted_np = np.clip(distorted_np, 0, 1)
        restored_np = np.clip(restored_np, 0, 1)

        dist_img = Image.fromarray((distorted_np * 255).astype(np.uint8))
        restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))

        dist_labeled = add_label(dist_img, "Distorted")
        restored_labeled = add_label(restored_img, "Restored")

        # ✅ 라벨 포함 이미지 결합
        if reference_exists:
            reference_np = reference_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            reference_np = np.clip(reference_np, 0, 1)
            ref_img = Image.fromarray((reference_np * 255).astype(np.uint8))
            ref_labeled = add_label(ref_img, "Reference")

            psnr_dist = compute_psnr(reference_np, distorted_np, data_range=1.0)
            ssim_dist = compute_ssim(reference_np, distorted_np, data_range=1.0, channel_axis=2)
            psnr_rest = compute_psnr(reference_np, restored_np, data_range=1.0)
            ssim_rest = compute_ssim(reference_np, restored_np, data_range=1.0, channel_axis=2)

            print(f"📌 {filename} | PSNR (Dist): {psnr_dist:.2f}, SSIM (Dist): {ssim_dist:.3f} | "
                  f"PSNR (Rest): {psnr_rest:.2f}, SSIM (Rest): {ssim_rest:.3f}")

            final = Image.new("RGB", (256 * 3, 280))
            final.paste(ref_labeled, (0, 0))
            final.paste(dist_labeled, (256, 0))
            final.paste(restored_labeled, (512, 0))
        else:
            print(f"⚠️ 참조 이미지 없음: {reference_path}")
            final = Image.new("RGB", (256 * 2, 280))
            final.paste(dist_labeled, (0, 0))
            final.paste(restored_labeled, (256, 0))

        final.save(save_path)

print(f"\n✅ CSIQ 전체 복원 완료. 결과 저장 경로: {SAVE_DIR}")
 """

# restore_rain100l
""" import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"
RAIN_ROOT = r"E:\restormer+volterra\data\rain100L"
SAVE_DIR = r"E:\restormer+volterra\results\restored_rain100L"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 라벨 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 함수
def process_split(split):
    rain_dir = os.path.join(RAIN_ROOT, split, "rain")
    norain_dir = os.path.join(RAIN_ROOT, split, "norain")
    save_subdir = os.path.join(SAVE_DIR, split)
    os.makedirs(save_subdir, exist_ok=True)

    for filename in os.listdir(rain_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or filename.startswith("._"):
            continue

        rain_path = os.path.join(rain_dir, filename)
        norain_path = os.path.join(norain_dir, filename)
        save_path = os.path.join(save_subdir, filename.replace(".", "_restored."))

        try:
            rain_pil = Image.open(rain_path).convert("RGB").resize((256, 256))
        except:
            print(f"⚠️ 이미지 열기 실패: {rain_path}")
            continue

        rain_tensor = transform(rain_pil).unsqueeze(0).to(DEVICE)
        has_ref = os.path.exists(norain_path)
        if has_ref:
            ref_pil = Image.open(norain_path).convert("RGB").resize((256, 256))
            ref_tensor = transform(ref_pil).unsqueeze(0).to(DEVICE)

        # 복원
        with torch.no_grad():
            with autocast():
                restored_tensor = model(rain_tensor)

        rain_np = rain_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        rain_np = np.clip(rain_np, 0, 1)
        restored_np = np.clip(restored_np, 0, 1)

        rain_img = Image.fromarray((rain_np * 255).astype(np.uint8))
        restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))
        rain_labeled = add_label(rain_img, "Distorted")
        restored_labeled = add_label(restored_img, "Restored")

        if has_ref:
            ref_np = ref_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            ref_np = np.clip(ref_np, 0, 1)
            ref_img = Image.fromarray((ref_np * 255).astype(np.uint8))
            ref_labeled = add_label(ref_img, "Reference")

            psnr_dist = compute_psnr(ref_np, rain_np, data_range=1.0)
            ssim_dist = compute_ssim(ref_np, rain_np, data_range=1.0, channel_axis=2)
            psnr_rest = compute_psnr(ref_np, restored_np, data_range=1.0)
            ssim_rest = compute_ssim(ref_np, restored_np, data_range=1.0, channel_axis=2)

            print(f"📌 [{split}] {filename} | PSNR(D): {psnr_dist:.2f}, SSIM(D): {ssim_dist:.3f} | "
                  f"PSNR(R): {psnr_rest:.2f}, SSIM(R): {ssim_rest:.3f}")

            final = Image.new("RGB", (256 * 3, 280))
            final.paste(ref_labeled, (0, 0))
            final.paste(rain_labeled, (256, 0))
            final.paste(restored_labeled, (512, 0))
        else:
            print(f"⚠️ 참조 이미지 없음: {norain_path}")
            final = Image.new("RGB", (256 * 2, 280))
            final.paste(rain_labeled, (0, 0))
            final.paste(restored_labeled, (256, 0))

        final.save(save_path)

# ✅ 실행
process_split("train")
process_split("test")

print(f"\n✅ Rain100L 전체 복원 완료. 결과 저장 경로: {SAVE_DIR}")
 """

# restore_HIDE
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from restormer_volterra import RestormerVolterra
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = r"E:\restormer+volterra\checkpoints\restormer_volterra_train_4sets\epoch_100.pth"
HIDE_ROOT = r"E:\restormer+volterra\data\HIDE"
SAVE_DIR = r"E:\restormer+volterra\results\restored_hide"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 라벨 함수
def add_label(img, text):
    labeled = Image.new("RGB", (256, 280), (255, 255, 255))
    labeled.paste(img, (0, 24))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    text_width = draw.textlength(text, font=font)
    draw.text(((256 - text_width) // 2, 4), text, fill=(0, 0, 0), font=font)
    return labeled

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 함수
def process_split(split):
    split_dir = os.path.join(HIDE_ROOT, split)
    gt_dir = os.path.join(HIDE_ROOT, "GT")
    save_subdir = os.path.join(SAVE_DIR, split)
    os.makedirs(save_subdir, exist_ok=True)

    for filename in os.listdir(split_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or filename.startswith("._"):
            continue

        distorted_path = os.path.join(split_dir, filename)
        gt_name = filename.replace("from", "from")  # 이름 그대로 대응
        gt_path = os.path.join(gt_dir, gt_name)
        save_path = os.path.join(save_subdir, filename.replace(".", "_restored."))

        try:
            distorted_pil = Image.open(distorted_path).convert("RGB").resize((256, 256))
        except:
            print(f"⚠️ 이미지 열기 실패: {distorted_path}")
            continue

        distorted_tensor = transform(distorted_pil).unsqueeze(0).to(DEVICE)
        has_ref = os.path.exists(gt_path)
        if has_ref:
            ref_pil = Image.open(gt_path).convert("RGB").resize((256, 256))
            ref_tensor = transform(ref_pil).unsqueeze(0).to(DEVICE)

        # ✅ 복원 수행
        with torch.no_grad():
            with autocast():
                restored_tensor = model(distorted_tensor)

        distorted_np = distorted_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        distorted_np = np.clip(distorted_np, 0, 1)
        restored_np = np.clip(restored_np, 0, 1)

        dist_img = Image.fromarray((distorted_np * 255).astype(np.uint8))
        restored_img = Image.fromarray((restored_np * 255).astype(np.uint8))

        dist_labeled = add_label(dist_img, "Distorted")
        restored_labeled = add_label(restored_img, "Restored")

        if has_ref:
            ref_np = ref_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            ref_np = np.clip(ref_np, 0, 1)
            ref_img = Image.fromarray((ref_np * 255).astype(np.uint8))
            ref_labeled = add_label(ref_img, "Reference")

            psnr_dist = compute_psnr(ref_np, distorted_np, data_range=1.0)
            ssim_dist = compute_ssim(ref_np, distorted_np, data_range=1.0, channel_axis=2)
            psnr_rest = compute_psnr(ref_np, restored_np, data_range=1.0)
            ssim_rest = compute_ssim(ref_np, restored_np, data_range=1.0, channel_axis=2)

            print(f"📌 [{split}] {filename} | PSNR(D): {psnr_dist:.2f}, SSIM(D): {ssim_dist:.3f} | "
                  f"PSNR(R): {psnr_rest:.2f}, SSIM(R): {ssim_rest:.3f}")

            final = Image.new("RGB", (256 * 3, 280))
            final.paste(ref_labeled, (0, 0))
            final.paste(dist_labeled, (256, 0))
            final.paste(restored_labeled, (512, 0))
        else:
            print(f"⚠️ 참조 이미지 없음: {gt_path}")
            final = Image.new("RGB", (256 * 2, 280))
            final.paste(dist_labeled, (0, 0))
            final.paste(restored_labeled, (256, 0))

        final.save(save_path)

# ✅ 실행
process_split("train")
process_split("test")

print(f"\n✅ HIDE 전체 복원 완료. 결과 저장 경로: {SAVE_DIR}")
