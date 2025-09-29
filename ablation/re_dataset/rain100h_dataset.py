# E:/restormer+volterra/ablation/re_dataset/rain100h_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import natsort


class Rain100HDataset(Dataset):
    def __init__(self, root_dir, transform=None, resize=(256, 256), debug=False):
        """
        Args:
            root_dir (str): Dataset root directory (e.g., E:/restormer+volterra/data/rain100H/train)
                            Must contain subfolders 'rain' and 'norain'.
            transform (callable, optional): Transformations to apply (default: Resize+ToTensor).
            resize (tuple, optional): Resize target size, default (256, 256).
            debug (bool): If True, print file matching info for debugging.
        """
        self.rain_dir = os.path.join(root_dir, "rain")
        self.norain_dir = os.path.join(root_dir, "norain")

        # 허용 확장자
        allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        # 파일 리스트 정렬
        self.rain_imgs = natsort.natsorted([
            f for f in os.listdir(self.rain_dir) if f.lower().endswith(allowed_ext)
        ])
        self.norain_imgs = natsort.natsorted([
            f for f in os.listdir(self.norain_dir) if f.lower().endswith(allowed_ext)
        ])

        assert len(self.rain_imgs) == len(self.norain_imgs), \
            f"Rain={len(self.rain_imgs)}, Norain={len(self.norain_imgs)} → 불일치!"

        # 기본 transform
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

        self.debug = debug

        print(f"[INFO] Loaded Rain100H dataset from {root_dir} | "
              f"Rain={len(self.rain_imgs)}, Norain={len(self.norain_imgs)}")

    def __len__(self):
        return len(self.rain_imgs)

    def __getitem__(self, idx):
        rain_path = os.path.join(self.rain_dir, self.rain_imgs[idx])
        norain_path = os.path.join(self.norain_dir, self.norain_imgs[idx])

        rain = Image.open(rain_path).convert("RGB")
        norain = Image.open(norain_path).convert("RGB")

        if self.transform:
            rain = self.transform(rain)
            norain = self.transform(norain)

        if self.debug:
            print(f"[{idx}] rain: {rain_path}")
            print(f"[{idx}] norain: {norain_path}")

        return rain, norain
