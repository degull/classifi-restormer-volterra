import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from slide_transformer import SlideTransformer

# ✅ 25개 distortion class 이름 정의 (KADID 기준)
class_names = [
    "gaussian_blur", "lens_blur", "motion_blur", "color_diffusion", "color_shift",
    "jpeg_compression", "jpeg2000_compression", "jpeg_xt", "jpegxr", "jpeg2000_transcode",
    "white_noise", "gaussian_noise", "impulse_noise", "quantization_noise", "multiplicative_noise",
    "brightness", "contrast", "underexposure", "overexposure", "colorfulness",
    "sharpness", "spatial_warping", "pixelate", "blocking", "non_eccentricity_pattern"
]

# ✅ 설정
MODEL_PATH = "E:/restormer+volterra/checkpoints/classifier_epoch199.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_distortion(image_path, return_softmax=False):
    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=25).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # ✅ 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # shape: [1, 3, 224, 224]

    # ✅ 예측
    with torch.no_grad():
        output = model(image_tensor, mode="infer")  # shape: [1, 25]
        probs = F.softmax(output, dim=1)
        pred_label = probs.argmax(dim=1).item()
        pred_class = class_names[pred_label]
        print(f"✅ 예측된 왜곡 유형: **{pred_class}** (label: {pred_label})")

    if return_softmax:
        return probs.squeeze(0).cpu().numpy()  # numpy 배열 반환
    else:
        return pred_label

# ✅ 단독 실행용 예시
if __name__ == "__main__":
    image_path = "C:/Users/IIPL02/Desktop/test_image.jpg"  # 예측할 이미지 경로 입력
    if os.path.exists(image_path):
        predict_distortion(image_path)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
