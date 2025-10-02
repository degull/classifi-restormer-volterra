import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import natsort

class CSDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        CSD Dataset 구조 예시:
        root_dir/
            Snow/
            Gt/
            Mask/
        """
        self.snow_dir = os.path.join(root_dir, "Snow")
        self.gt_dir   = os.path.join(root_dir, "Gt")

        self.snow_imgs = natsort.natsorted(os.listdir(self.snow_dir))
        self.gt_imgs   = natsort.natsorted(os.listdir(self.gt_dir))

        assert len(self.snow_imgs) == len(self.gt_imgs), \
            f"개수 불일치: Snow={len(self.snow_imgs)}, Gt={len(self.gt_imgs)}"

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        print(f"[INFO] Loaded CSD dataset from {root_dir} | Snow={len(self.snow_imgs)}, Gt={len(self.gt_imgs)}")

    def __len__(self):
        return len(self.snow_imgs)

    def __getitem__(self, idx):
        snow_path = os.path.join(self.snow_dir, self.snow_imgs[idx])
        gt_path   = os.path.join(self.gt_dir,   self.gt_imgs[idx])

        snow = Image.open(snow_path).convert("RGB")
        gt   = Image.open(gt_path).convert("RGB")

        return self.transform(snow), self.transform(gt)
