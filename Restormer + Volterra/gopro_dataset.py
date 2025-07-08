# 예시: re_dataset/gopro_dataset.py
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

class GoProDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blur_path = self.data.iloc[idx]["blur"]
        sharp_path = self.data.iloc[idx]["sharp"]

        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img
