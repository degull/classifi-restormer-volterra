# e:/restormer+volterra/re_dataset/rain100l_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import natsort

class Rain100LDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.rain_dir = os.path.join(root_dir, "rain")
        self.norain_dir = os.path.join(root_dir, "norain")

        self.rain_images = natsort.natsorted([
            f for f in os.listdir(self.rain_dir) if f.lower().endswith(('.png', '.jpg'))
        ])
        self.norain_images = natsort.natsorted([
            f for f in os.listdir(self.norain_dir) if f.lower().endswith(('.png', '.jpg'))
        ])

        assert len(self.rain_images) == len(self.norain_images), "Rain and NoRain counts must match"

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, idx):
        rain_path = os.path.join(self.rain_dir, self.rain_images[idx])
        clean_path = os.path.join(self.norain_dir, self.norain_images[idx])

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        rain_tensor = self.transform(rain_img)
        clean_tensor = self.transform(clean_img)

        return rain_tensor, clean_tensor
