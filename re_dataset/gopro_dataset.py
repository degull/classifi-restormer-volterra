import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GoProDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (str): CSV 경로 (gopro_train_pairs.csv 또는 gopro_test_pairs.csv)
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        blur_path = row['dist_img']
        sharp_path = row['ref_img']

        # 🔍 경로 출력
        print(f"[{idx}] blur: {blur_path}")
        print(f"[{idx}] sharp: {sharp_path}")

        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')

        # 🔍 원본 이미지 사이즈 출력
        print(f"[{idx}] blur size: {blur.size}, sharp size: {sharp.size}")

        blur = self.transform(blur)
        sharp = self.transform(sharp)

        # 🔍 텐서 shape 출력
        print(f"[{idx}] blur tensor shape: {blur.shape}, sharp tensor shape: {sharp.shape}")

        return blur, sharp
