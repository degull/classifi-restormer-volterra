# E:\MRVNet2D\Restormer + Volterra\train.py
#train.py
""" import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from restormer_volterra import RestormerVolterra  # 모델 정의
from kadid_dataset import KADID10KDataset  # 커스텀 데이터셋
from torch.cuda.amp import autocast, GradScaler  # ✅ AMP 모듈

# ✅ 학습 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
DATA_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
DISTORTED_DIR = 'E:/MRVNet2D/dataset/KADID10K/images'
SAVE_DIR = 'checkpoints/restormer_volterra_kadid'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def main():
    # ✅ 데이터 로더 정의
    train_dataset = KADID10KDataset(csv_file=DATA_CSV, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ✅ 모델, 손실함수, 옵티마이저
    model = RestormerVolterra().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ✅ AMP 초기화
    scaler = GradScaler()

    # ✅ 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for distorted, reference in loop:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)

            optimizer.zero_grad()

            with autocast():  # ✅ 혼합 정밀도 적용
                output = model(distorted)
                loss = criterion(output, reference)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.6f}")

        # ✅ 모든 에포크에서 저장
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    main()

 """

# E:\MRVNet2D\Restormer + Volterra\train.py
#train.py
# Progressive Learning

 
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
from tid_dataset import TID2013Dataset
from csiq_dataset import CSIQDataset

# ✅ 학습 설정
BATCH_SIZE = 2
EPOCHS = 100
LR = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ 경로 설정
KADID_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
TID_CSV = 'E:/MRVNet2D/dataset/tid2013/mos.csv'
TID_DISTORTED_DIR = 'E:/MRVNet2D/dataset/tid2013/distorted_images'
TID_REFERENCE_DIR = 'E:/MRVNet2D/dataset/tid2013/reference_images'
CSIQ_CSV = 'E:/MRVNet2D/dataset/CSIQ/CSIQ.txt'
CSIQ_ROOT = 'E:/MRVNet2D/dataset/CSIQ'

SAVE_DIR = 'checkpoints/restormer_volterra_all_60'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Progressive Learning 스케줄
resize_schedule = {
    0: 128,
    30: 192,
    60: 256  # 학습은 60까지
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
        tid_dataset = TID2013Dataset(csv_file=TID_CSV,
                                      distorted_dir=TID_DISTORTED_DIR,
                                      reference_dir=TID_REFERENCE_DIR,
                                      transform=transform)
        csiq_dataset = CSIQDataset(csv_file=CSIQ_CSV,
                                    root_dir=CSIQ_ROOT,
                                    transform=transform)

        # ✅ 데이터셋 통합
        train_dataset = ConcatDataset([kadid_dataset, tid_dataset, csiq_dataset])
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
