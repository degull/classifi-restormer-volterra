import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# class = 25(왜곡분류)
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from model.slide_transformer import SlideTransformer

# ✅ distortion index -> name 매핑
distortion_mapping = {
    0: "Gaussian blur",
    1: "Lens blur",
    2: "Motion blur",
    3: "Color diffusion",
    4: "Color shift",
    5: "Color quantization",
    6: "Color saturation 1",
    7: "Color saturation 2",
    8: "JPEG2000 compression",
    9: "JPEG compression",
    10: "White noise",
    11: "White noise in YCbCr",
    12: "Impulse noise",
    13: "Multiplicative noise",
    14: "Denoise (DnCNN)",
    15: "Brighten",
    16: "Darken",
    17: "Mean shift",
    18: "Jitter",
    19: "Non-eccentricity patch",
    20: "Pixelate",
    21: "Quantization",
    22: "Color block",
    23: "High sharpen",
    24: "Contrast change"
}

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/class_25_kadid/final_DAS-Transformer_KADID10K.pth"

# ✅ 모델 로드
model = SlideTransformer(img_size=224, num_classes=25).to(DEVICE)
# ✅ 이걸로 변경
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

model.eval()

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ 테스트할 이미지 경로
test_img_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images/I81_11_04.png"  # <- 여기에 테스트할 이미지 넣기

# ✅ 단일 이미지 로드 및 예측
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    with torch.no_grad():
        output = model(image, mode="infer")
        pred_label = output.argmax(dim=1).item()

    distortion_name = distortion_mapping[pred_label]
    return pred_label, distortion_name

# ✅ 실행
if __name__ == "__main__":
    pred_label, distortion_name = predict_image(test_img_path)

    print(f"✅ 예측된 클래스 인덱스: {pred_label}")
    print(f"✅ 예측된 왜곡 종류: {distortion_name}")
