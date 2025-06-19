# E:\MRVNet2D\Restormer + Volterra\train.py
#train.py
# E:/MRVNet2D/Restormer + Volterra/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from restormer_volterra import RestormerVolterra
from kadid_dataset import KADID10KDataset
from rain100h_dataset import Rain100HDataset
from gopro_dataset import GoProDataset
from sidd_dataset import SIDDDataset

# ✅ 학습 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
KADID_CSV = 'E:/restormer+volterra/data/KADID10K/kadid10k.csv'
RAIN100H_DIR = 'E:/restormer+volterra/data/rain100H/train'
GOPRO_CSV = 'E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv'
SIDD_DIR = 'E:/restormer+volterra/data/SIDD/train'

SAVE_DIR = 'checkpoints/restormer_volterra_train_4sets'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning 스케줄
resize_schedule = {
    0: 128,
    30: 192,
    60: 256
}

def get_transform(epoch):
    size = 256
    for key in sorted(resize_schedule.keys()):
        if epoch >= key:
            size = resize_schedule[key]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def main():
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        transform = get_transform(epoch)

        # ✅ 데이터셋 로드
        kadid_dataset = KADID10KDataset(csv_file=KADID_CSV, transform=transform)
        rain100h_dataset = Rain100HDataset(root_dir=RAIN100H_DIR, transform=transform)
        gopro_dataset = GoProDataset(csv_path=GOPRO_CSV, transform=transform)
        sidd_dataset = SIDDDataset(root_dir=SIDD_DIR, transform=transform)

        # ✅ 데이터셋 통합
        train_dataset = ConcatDataset([
            kadid_dataset,
            rain100h_dataset,
            gopro_dataset,
            sidd_dataset
        ])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)

        print(f"[Epoch {epoch+1}] Input resolution: {transform.transforms[0].size}, Total samples: {len(train_dataset)}")

        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.6f}")

        # ✅ 모델 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()
