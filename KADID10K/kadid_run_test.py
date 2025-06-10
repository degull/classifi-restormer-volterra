# (class = 7) -> 이거 사용함

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append("C:/Users/IIPL02/Desktop/NEW")
import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/kadid/7class_DAS-Transformer_epoch_199.pth"

BATCH_SIZE = 32
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ 테스트 시작
if __name__ == '__main__':
    test_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for dist_img, labels in test_loader:
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)
            outputs = model(dist_img)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # ✅ Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_names = [
        "blur",
        "color_distortion",
        "compression",
        "noise",
        "brightness_change",
        "spatial_distortion",
        "sharpness_contrast"
    ]

    # ✅ 정확도 비율 계산 및 출력
    print("\n📊 정확도 분석 (Confusion Matrix 기반)")
    total_correct = 0
    total_samples = 0

    for i, class_name in enumerate(class_names):
        correct = conf_matrix[i, i]
        total = conf_matrix[i].sum()
        acc = correct / total * 100
        total_correct += correct
        total_samples += total
        print(f"{class_name:>20s} ({acc:.2f}%): {total}개 중 {correct}개 예측")

    overall_acc = total_correct / total_samples * 100
    print(f"\n🎯 전체 정확도: {overall_acc:.2f}%")

    # ✅ SRCC & PLCC
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    print(f"\n📌 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    # ✅ Classification Report
    print("\n📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    # ✅ 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of DAS-Transformer on KADID-10k (7-Class)")
    plt.tight_layout()
    plt.show()
