
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr

from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
CHECKPOINT_DIR = "C:/Users/IIPL02/Desktop/NEW/checkpoints/hierarchical_kadid"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# ✅ coarse class 매핑 (공식 기준)
COARSE_CLASS_MAP = {
    0: list(range(0, 3)),    # Blurs
    1: list(range(3, 8)),    # Color distortions
    2: list(range(8, 10)),   # Compression
    3: list(range(10, 15)),  # Noise
    4: list(range(15, 18)),  # Brightness change
    5: list(range(18, 23)),  # Spatial distortions
    6: list(range(23, 25))   # Sharpness & contrast
}
NUM_COARSE_CLASSES = len(COARSE_CLASS_MAP)
MAX_FINE_CLASSES = max(len(v) for v in COARSE_CLASS_MAP.values())

# ✅ 전처리
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

def get_fine_local_label(fine_label):
    for fine_list in COARSE_CLASS_MAP.values():
        if fine_label in fine_list:
            return fine_list.index(fine_label)
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

def train():
    dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = HierarchicalClassifier(NUM_COARSE_CLASSES, MAX_FINE_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total = 0
        correct_coarse = 0
        correct_fine = 0

        for batch_idx, (imgs, fine_labels) in enumerate(dataloader):
            imgs, fine_labels = imgs.to(DEVICE), fine_labels.to(DEVICE)
            coarse_labels = torch.tensor([get_coarse_label(l.item()) for l in fine_labels], device=DEVICE)
            fine_local_labels = torch.tensor([get_fine_local_label(l.item()) for l in fine_labels], device=DEVICE)

            optimizer.zero_grad()
            coarse_logits, fine_logits, _ = model(imgs, mode="train")
            loss = criterion(coarse_logits, coarse_labels) + criterion(fine_logits, fine_local_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_coarse += (coarse_logits.argmax(1) == coarse_labels).sum().item()
            correct_fine += (fine_logits.argmax(1) == fine_local_labels).sum().item()
            total += fine_labels.size(0)

        scheduler.step()
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss: {total_loss:.4f} | "
              f"Coarse Acc: {correct_coarse/total:.4f} | Fine Acc: {correct_fine/total:.4f}")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train()
