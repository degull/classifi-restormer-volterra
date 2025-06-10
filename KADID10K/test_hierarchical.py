import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer
from torch import nn

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/hierarchical_kadid/epoch_100.pth"

# ✅ coarse class 매핑
COARSE_CLASS_MAP = {
    0: list(range(0, 3)),    # Blurs
    1: list(range(3, 8)),    # Color distortions
    2: list(range(8, 10)),   # Compression
    3: list(range(10, 15)),  # Noise
    4: list(range(15, 18)),  # Brightness change
    5: list(range(18, 23)),  # Spatial distortions
    6: list(range(23, 25))   # Sharpness & contrast
}

coarse_class_names = [
    "Blur", "Color", "Compression", "Noise",
    "Brightness", "Spatial", "Sharp/Contrast"
]

fine_class_names = [
    "Gaussian blur", "Lens blur", "Motion blur",
    "Color diffusion", "Color shift", "Color quantization",
    "Color saturation 1", "Color saturation 2",
    "JPEG2000", "JPEG",
    "White noise", "YCbCr noise", "Impulse noise",
    "Multiplicative noise", "Denoised (DnCNN)",
    "Brighten", "Darken", "Mean shift",
    "Jitter", "Non-eccentricity", "Pixelate",
    "Quantization", "Color block",
    "High sharpen", "Contrast change"
]

NUM_COARSE_CLASSES = len(COARSE_CLASS_MAP)
MAX_FINE_CLASSES = max(len(v) for v in COARSE_CLASS_MAP.values())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_coarse_label(fine_label):
    for c_idx, fine_list in COARSE_CLASS_MAP.items():
        if fine_label in fine_list:
            return c_idx
    return -1

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_coarse, max_fine_classes):
        super().__init__()
        self.coarse_head = SlideTransformer(img_size=224, num_classes=num_coarse)
        self.fine_heads = nn.ModuleList([
            SlideTransformer(img_size=224, num_classes=max_fine_classes)
            for _ in range(num_coarse)
        ])

    def forward(self, x, mode="train"):
        coarse_logits = self.coarse_head(x, mode)
        coarse_preds = torch.argmax(coarse_logits, dim=1)
        fine_logits_batch = []
        for i in range(x.size(0)):
            c = coarse_preds[i].item()
            fine_logits = self.fine_heads[c](x[i].unsqueeze(0), mode)
            fine_logits_batch.append(fine_logits)
        fine_logits = torch.cat(fine_logits_batch, dim=0)
        return coarse_logits, fine_logits, coarse_preds

def run_test():
    dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = HierarchicalClassifier(NUM_COARSE_CLASSES, MAX_FINE_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_coarse = []
    pred_coarse = []
    true_fine = []
    pred_fine = []

    with torch.no_grad():
        for imgs, fine_labels in dataloader:
            imgs = imgs.to(DEVICE)
            fine_labels = fine_labels.to(DEVICE)
            coarse_labels = torch.tensor([get_coarse_label(l.item()) for l in fine_labels], device=DEVICE)

            coarse_logits, fine_logits, coarse_preds = model(imgs, mode="infer")

            true_coarse.extend(coarse_labels.cpu().numpy())
            pred_coarse.extend(coarse_preds.cpu().numpy())
            true_fine.extend(fine_labels.cpu().numpy())
            pred_fine.extend(fine_logits.argmax(1).cpu().numpy())

    # ✅ Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(true_coarse, pred_coarse), annot=True, fmt="d",
                xticklabels=coarse_class_names, yticklabels=coarse_class_names, cmap="Blues")
    plt.title("Coarse Class Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 10))
    sns.heatmap(confusion_matrix(true_fine, pred_fine), annot=False, cmap="Blues",
                xticklabels=fine_class_names, yticklabels=fine_class_names)
    plt.title("Fine Class Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ✅ Classification Report
    print("📌 Coarse Classification Report:")
    print(classification_report(true_coarse, pred_coarse, target_names=coarse_class_names))

    print("📌 Fine Classification Report:")
    print(classification_report(true_fine, pred_fine, target_names=fine_class_names))

if __name__ == "__main__":
    run_test()
