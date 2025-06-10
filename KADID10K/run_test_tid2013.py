# run_test_tid2013.py
import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_tid2013 import TID2013Dataset
from model.slide_transformer import SlideTransformer

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/TID2013/tid2013.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/TID2013"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/7class_DAS-Transformer_epoch_199.pth"

BATCH_SIZE = 32
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    test_dataset = TID2013Dataset(CSV_PATH, IMG_DIR, transform=transform)
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

    # ✅ 결과 출력
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_names = [
        "blur", "color_distortion", "compression", "noise",
        "brightness_change", "spatial_distortion", "sharpness_contrast"
    ]

    print("\n📊 정확도 분석 (Confusion Matrix 기반)")
    for i, name in enumerate(class_names):
        correct = conf_matrix[i, i]
        total = conf_matrix[i].sum()
        acc = correct / total * 100 if total > 0 else 0
        print(f"{name:>20s} ({acc:.2f}%): {total}개 중 {correct}개 예측")

    overall_acc = np.trace(conf_matrix) / np.sum(conf_matrix) * 100
    print(f"\n🎯 전체 정확도: {overall_acc:.2f}%")

    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    print(f"\n📌 SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    print("\n📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (TID2013 → KADID-trained model)")
    plt.tight_layout()
    plt.show()
