import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from slide_transformer import SlideTransformer
from restormer_volterra import RestormerVolterra

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_PATH = "E:/restormer+volterra/test_images/test_blurjpeg.png"   # 테스트 이미지 경로
SAVE_PATH = "E:/restormer+volterra/results/restored_output.png"    # 저장 경로

CLASSIFIER_WEIGHTS = "E:/restormer+volterra/checkpoints/classifier_epoch199.pth"
RESTORER_WEIGHTS = "E:/restormer+volterra/checkpoints/restorer_epoch100.pth"

# ✅ 이미지 전처리 (224x224로 resize 및 정규화)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ 이미지 로드 함수
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

# ✅ 결과 저장 함수
def save_image(tensor, save_path):
    tensor = (tensor.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    tensor = tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(tensor)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_image.save(save_path)
    print(f"✅ 복원 이미지 저장 완료: {save_path}")

# ✅ 분류기 로드
classifier = SlideTransformer(img_size=224, num_classes=25).to(DEVICE)
classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
classifier.eval()

# ✅ 복원기 로드
restorer = RestormerVolterra(cond_dim=25).to(DEVICE)
restorer.load_state_dict(torch.load(RESTORER_WEIGHTS, map_location=DEVICE))
restorer.eval()

# ✅ 1. 이미지 로드
input_tensor = load_image(IMG_PATH).to(DEVICE)  # shape: [1, 3, 224, 224]

# ✅ 2. SlideTransformer로 condition vector 얻기
with torch.no_grad():
    logits = classifier(input_tensor, mode="infer")  # shape: [1, 25]
    condition_vector = F.softmax(logits, dim=1)      # 확률 벡터로 변환

# ✅ 3. RestormerVolterra로 adaptive 복원
with torch.no_grad():
    restored = restorer(input_tensor, condition_vector)  # shape: [1, 3, 224, 224]

# ✅ 4. 결과 저장
save_image(restored, SAVE_PATH)
