# tid2013_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class TID2013Dataset(Dataset):
    def __init__(self, csv_file, distorted_dir, reference_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.distorted_dir = distorted_dir
        self.reference_dir = reference_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distorted_name = self.data.iloc[idx, 0]
        ref_id = distorted_name.split('_')[0].upper()  # 'i01_02_5.bmp' → 'I01.BMP'
        reference_name = f"{ref_id}.BMP"

        distorted_path = os.path.join(self.distorted_dir, distorted_name)
        reference_path = os.path.join(self.reference_dir, reference_name)

        distorted = Image.open(distorted_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")

        if self.transform:
            distorted = self.transform(distorted)
            reference = self.transform(reference)

        return distorted, reference
