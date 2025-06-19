# E:\restormer+volterra\classi+restormer+volterra\train_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from slide_transformer import SlideTransformer

# ✅ KADID10K 분류용 Dataset (distorted + class from filename)
class KADID10KClassifierDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx]["dist_img"]  # e.g., I01_01_01.png
        distortion_code = int(filename.split("_")[1]) - 1  # 01 -> 0, ..., 25 -> 24
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, distortion_code


def main():
    # ✅ 설정
    CSV_PATH = "E:/restormer+volterra/data/KADID10K/kadid10k.csv"
    IMAGE_DIR = "E:/restormer+volterra/data/KADID10K/images"
    CHECKPOINT_DIR = "E:/restormer+volterra/checkpoints/classifier"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 32
    TOTAL_EPOCHS = 150
    RESUME_EPOCH = 72
    LR = 1e-4
    NUM_CLASSES = 25
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 데이터셋 및 DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dataset = KADID10KClassifierDataset(CSV_PATH, IMAGE_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # ✅ 모델, 손실 함수, 최적화 도구
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ✅ 이어서 학습할 경우 체크포인트 로드
    resume_path = os.path.join(CHECKPOINT_DIR, f"classifier_epoch{RESUME_EPOCH}.pth")
    if os.path.exists(resume_path):
        print(f"🔁 이전 가중치 불러오는 중: {resume_path}")
        checkpoint = torch.load(resume_path)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"⚠️ strict=True 로드 실패: {e}")
            print("🔁 strict=False 로 로드 시도")
            model.load_state_dict(checkpoint, strict=False)
        print("✅ 모델 가중치 로드 완료")

    # ✅ 학습 루프
    for epoch in range(RESUME_EPOCH + 1, TOTAL_EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}/{TOTAL_EPOCHS}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, mode="train")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            acc = 100. * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        print(f"✅ Epoch {epoch}: Loss = {epoch_loss:.4f}, Acc = {epoch_acc:.2f}%")

        scheduler.step()

        # ✅ 모델 저장
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"classifier_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("🎉 분류기 이어서 학습 완료!")


# ✅ Windows-safe entry point
if __name__ == "__main__":
    main()

# ✅ Epoch 139: Loss = 0.3504, Acc = 85.68%