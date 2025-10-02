import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GoProDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file: gopro_train_pairs.csv or gopro_test_pairs.csv
        CSV 형식 (헤더 있음):
            dist_img,ref_img
            abs_or_rel_path_to_blur.png, abs_or_rel_path_to_sharp.png
        """
        self.root_dir = os.path.dirname(csv_file)
        # ✅ 헤더를 무시하지 않고 읽기
        self.pairs = pd.read_csv(csv_file)  # header=0 (기본값)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path = str(self.pairs.iloc[idx, 0]).strip()
        sharp_path = str(self.pairs.iloc[idx, 1]).strip()

        # ✅ 절대경로가 아니면 root_dir 기준으로 처리
        if not os.path.isabs(blur_path):
            blur_path = os.path.join(self.root_dir, blur_path)
        if not os.path.isabs(sharp_path):
            sharp_path = os.path.join(self.root_dir, sharp_path)

        if not os.path.exists(blur_path):
            raise FileNotFoundError(f"[❌] Blur image not found: {blur_path}")
        if not os.path.exists(sharp_path):
            raise FileNotFoundError(f"[❌] Sharp image not found: {sharp_path}")

        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')

        return self.transform(blur), self.transform(sharp)
