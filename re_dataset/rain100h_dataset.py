import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import natsort

class Rain100HDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): e.g., E:/restormer+volterra/data/rain100H
            split (str): 'train' or 'test'
        """
        self.rain_dir = os.path.join(root_dir, "rain")
        self.norain_dir = os.path.join(root_dir, "norain")


        self.rain_imgs = natsort.natsorted(os.listdir(self.rain_dir))
        self.norain_imgs = natsort.natsorted(os.listdir(self.norain_dir))

        assert len(self.rain_imgs) == len(self.norain_imgs), "Rain and NoRain image counts must match"

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.rain_imgs)

    def __getitem__(self, idx):
        rain_path = os.path.join(self.rain_dir, self.rain_imgs[idx])
        norain_path = os.path.join(self.norain_dir, self.norain_imgs[idx])

        rain = Image.open(rain_path).convert('RGB')
        norain = Image.open(norain_path).convert('RGB')

        rain = self.transform(rain)
        norain = self.transform(norain)

        # 🔍 디버깅용
        print(f"[{idx}] rain: {rain_path}")
        print(f"[{idx}] norain: {norain_path}")

        return rain, norain
