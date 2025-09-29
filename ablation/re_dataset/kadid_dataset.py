import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class KadidDataset(Dataset):
    def __init__(self, csv_path, image_dir, split_path, split='train', transform=None):
        """
        Args:
            csv_path (str): e.g., E:/restormer+volterra/data/KADID10K/kadid10k.csv
            image_dir (str): e.g., E:/restormer+volterra/data/KADID10K/images
            split_path (str): e.g., E:/restormer+volterra/data/KADID10K/splits/
            split (str): 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 🔹 split 로드
        split_file = os.path.join(split_path, f"{split}.npy")
        self.split_indices = np.load(split_file)

        # 🔹 split에 해당하는 행만 필터링
        self.data = self.data.iloc[self.split_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.image_dir, row['dist_img'])  # 상대 경로
        ref_img_path = os.path.join(self.image_dir, row['ref_img'])    # 상대 경로

        # 🔍 디버깅 출력
        print(f"[{idx}] dist: {dist_img_path}")
        print(f"[{idx}] ref: {ref_img_path}")

        dist = Image.open(dist_img_path).convert('RGB')
        ref = Image.open(ref_img_path).convert('RGB')

        dist = self.transform(dist)
        ref = self.transform(ref)

        return dist, ref
