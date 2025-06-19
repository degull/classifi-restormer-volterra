# E:\restormer+volterra\classi+restormer+volterra\dataset_kadid10k.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ✅ 공통 이미지 전처리 (224x224, [-1,1])
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


# ✅ KADID10K 분류용 Dataset (distorted + label)
class KADID10KClassifierDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir  # e.g., E:/restormer+volterra/data/KADID10K/images
        self.transform = transform if transform else get_default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ✅ dist_img: e.g., I01_01_01.png
        filename = self.data.iloc[idx]["dist_img"]

        # ✅ distortion class 추출: filename = I01_**01**_01.png → 중간 segment가 distortion type
        distortion_code = int(filename.split("_")[1]) - 1  # "01" → 0, ..., "25" → 24

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, distortion_code



# ✅ KADID10K 복원용 Dataset (distorted + reference)
class KADID10KRestorationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform if transform else get_default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_path = self.data.iloc[idx]["dis_img_path"]
        ref_path = self.data.iloc[idx]["ref_img_path"]

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        return self.transform(dist_img), self.transform(ref_img)


# ✅ TID2013 복원용 Dataset
class TID2013RestorationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform if transform else get_default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_path = self.data.iloc[idx]["dis_img_path"]
        ref_path = self.data.iloc[idx]["ref_img_path"]

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        return self.transform(dist_img), self.transform(ref_img)


# ✅ CSIQ 복원용 Dataset
class CSIQRestorationDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.data = pd.read_csv(txt_path, sep=",")  # dis_img_path, dis_type, ref_img_path, score
        self.transform = transform if transform else get_default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_path = self.data.iloc[idx]["dis_img_path"]
        ref_path = self.data.iloc[idx]["ref_img_path"]

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        return self.transform(dist_img), self.transform(ref_img)
