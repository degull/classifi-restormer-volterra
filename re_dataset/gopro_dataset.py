import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class GoProDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.pairs = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                blur_path, sharp_path = row
                self.pairs.append((blur_path, sharp_path))

        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')
        return self.transform(blur), self.transform(sharp)
