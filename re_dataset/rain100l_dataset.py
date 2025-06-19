# rain100l_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Rain100LDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        root_dir: 루트 디렉토리 (예: E:/restormer+volterra/data/rain100L)
        mode: 'train' 또는 'test'
        transform: torchvision.transforms
        """
        self.rain_dir = os.path.join(root_dir, mode, 'rain')
        self.norain_dir = os.path.join(root_dir, mode, 'norain')

        self.rain_images = sorted([
            os.path.join(self.rain_dir, fname)
            for fname in os.listdir(self.rain_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.norain_images = sorted([
            os.path.join(self.norain_dir, fname)
            for fname in os.listdir(self.norain_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        assert len(self.rain_images) == len(self.norain_images), \
            "비 이미지와 정답 이미지 수가 다릅니다."

        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, idx):
        rain_image = Image.open(self.rain_images[idx]).convert("RGB")
        clean_image = Image.open(self.norain_images[idx]).convert("RGB")

        return {
            'rain': self.transform(rain_image),
            'clean': self.transform(clean_image),
            'rain_path': self.rain_images[idx],
            'clean_path': self.norain_images[idx]
        }
