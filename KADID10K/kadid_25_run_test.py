import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer
from tqdm import tqdm

# ✅ distortion index → 이름 매핑
distortion_mapping = {
    0: "Gaussian blur", 1: "Lens blur", 2: "Motion blur",
    3: "Color diffusion", 4: "Color shift", 5: "Color quantization",
    6: "Color saturation 1", 7: "Color saturation 2",
    8: "JPEG2000 compression", 9: "JPEG compression",
    10: "White noise", 11: "White noise in YCbCr",
    12: "Impulse noise", 13: "Multiplicative noise",
    14: "Denoise (DnCNN)", 15: "Brighten", 16: "Darken",
    17: "Mean shift", 18: "Jitter", 19: "Non-eccentricity patch",
    20: "Pixelate", 21: "Quantization", 22: "Color block",
    23: "High sharpen", 24: "Contrast change"
}

def main():
    # ✅ 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
    MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/class_25_kadid/epoch_197.pth"
    BATCH_SIZE = 64

    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=25).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    # ✅ 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ✅ 평가 시작
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, mode="infer")
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ✅ Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ✅ Classification Report 출력
    report = classification_report(all_labels, all_preds, target_names=[distortion_mapping[i] for i in range(25)])
    print("\n📄 Classification Report:\n")
    print(report)

    # ✅ Confusion Matrix 시각화
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[distortion_mapping[i] for i in range(25)],
                yticklabels=[distortion_mapping[i] for i in range(25)])
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("C:/Users/IIPL02/Desktop/NEW/confusion_matrix_kadid25.png")
    plt.show()

    # ✅ 클래스별 정확도 출력
    print("\n✅ 클래스별 정확도:")
    for i in range(25):
        correct = (np.array(all_preds) == np.array(all_labels)) & (np.array(all_labels) == i)
        total = (np.array(all_labels) == i)
        acc = correct.sum() / total.sum()
        print(f"{distortion_mapping[i]}: {acc * 100:.2f}%")

# ✅ Windows에서는 반드시 이렇게
if __name__ == "__main__":
    main()
