import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from slide_transformer import SlideTransformer

# ✅ 클래스 이름 (class=7 기준)
class_names = [
    "blur",
    "color_distortion",
    "compression",
    "noise",
    "brightness_change",
    "spatial_distortion",
    "sharpness_contrast"
]

# ✅ 설정
MODEL_PATH = "E:\restormer+volterra\checkpoints\kadid\7class_DAS-Transformer_epoch_199.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_distortion(image_path):
    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # ✅ 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    # ✅ 예측
    with torch.no_grad():
        output = model(image)
        pred_label = output.argmax(dim=1).item()

    # ✅ 결과 출력
    print(f"✅ 예측된 왜곡 유형: **{class_names[pred_label]}** (label: {pred_label})")

# ✅ 실행 예시
if __name__ == "__main__":
    image_path = "C:/Users/IIPL02/Desktop/test_image.jpg"  # 예측할 이미지 경로 입력
    if os.path.exists(image_path):
        predict_distortion(image_path)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
