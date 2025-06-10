# 왜곡 분류 세분화(# class=7)
""" import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ 25개 왜곡을 7개 그룹으로 정리 (공식 문서 기준)
        self.distortion_groups = {
            "blur": ["01", "02", "03"],  # Gaussian Blur, Lens Blur, Motion Blur
            "color_distortion": ["04", "05", "06", "07", "08"],  # Color 관련 왜곡
            "compression": ["09", "10"],  # JPEG, JPEG2000
            "noise": ["11", "12", "13", "14", "15"],  # White Noise, Impulse Noise, Denoise 등
            "brightness_change": ["16", "17", "18"],  # 밝기 조정
            "spatial_distortion": ["19", "20", "21", "22", "23"],  # Jitter, Pixelate 등
            "sharpness_contrast": ["24", "25"],  # High Sharpen, Contrast Change
        }

    def get_distortion_group(self, dist_img):
        distortion_type = dist_img.split("_")[1]  # 예: 'I01_01_01.png' → '01'
        for group, codes in self.distortion_groups.items():
            if distortion_type in codes:
                return group
        return "unknown"  # 알 수 없는 경우

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.img_dir, row["dist_img"])

        # ✅ 이미지 로드
        dist_img = Image.open(dist_img_path).convert("RGB")

        # ✅ 변환 적용
        if self.transform:
            dist_img = self.transform(dist_img)

        # ✅ 왜곡 그룹 라벨
        distortion_group = self.get_distortion_group(row["dist_img"])
        label = list(self.distortion_groups.keys()).index(distortion_group)  # 정수 라벨 변환

        return dist_img, label

# ✅ 데이터셋 테스트 실행
if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = KADID10KDataset(csv_path, img_dir, transform=transform)

    # ✅ 샘플 데이터 확인
    dist_img, label = dataset[100]
    print(f"✅ Distorted Image Shape: {dist_img.shape}")
    print(f"✅ Distortion Label (Group Index): {label}")
    print(f"✅ Distortion Group: {list(dataset.distortion_groups.keys())[label]}")
 """



""" xx_yy_zz.png:

xx: 원본 이미지 ID
yy: 왜곡 코드 (01~25)
zz: 왜곡 레벨 (1~5) → 이번 프로젝트에서는 고려하지 않음 """

# ref 제외
""" import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ ref 이미지가 dist_img에 들어가지 않도록 필터링
        self.data = self.data[self.data['dist_img'].str.contains("_")]  # ref는 "_" 없음

        # ✅ 25개 왜곡을 7개 그룹으로 정리 (공식 문서 기준)
        self.distortion_groups = {
            "blur": ["01", "02", "03"],  # Gaussian Blur, Lens Blur, Motion Blur
            "color_distortion": ["04", "05", "06", "07", "08"],  # Color 관련 왜곡
            "compression": ["09", "10"],  # JPEG, JPEG2000
            "noise": ["11", "12", "13", "14", "15"],  # White Noise, Impulse Noise, Denoise 등
            "brightness_change": ["16", "17", "18"],  # 밝기 조정
            "spatial_distortion": ["19", "20", "21", "22", "23"],  # Jitter, Pixelate 등
            "sharpness_contrast": ["24", "25"],  # High Sharpen, Contrast Change
        }

    def get_distortion_group(self, dist_img):
        distortion_type = dist_img.split("_")[1]  # 예: 'I01_01_01.png' → '01'
        for group, codes in self.distortion_groups.items():
            if distortion_type in codes:
                return group
        return "unknown"  # 알 수 없는 경우

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_img_path = os.path.join(self.img_dir, row["dist_img"])  # ✅ dist_img만 사용

        # ✅ 이미지 로드
        dist_img = Image.open(dist_img_path).convert("RGB")

        # ✅ 변환 적용
        if self.transform:
            dist_img = self.transform(dist_img)

        # ✅ 왜곡 그룹 라벨
        distortion_group = self.get_distortion_group(row["dist_img"])
        label = list(self.distortion_groups.keys()).index(distortion_group)

        return dist_img, label

# ✅ 샘플 테스트
if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = KADID10KDataset(csv_path, img_dir, transform=transform)

    dist_img, label = dataset[100]
    print(f"✅ Distorted Image Shape: {dist_img.shape}")
    print(f"✅ Distortion Label (Group Index): {label}")
    print(f"✅ Distortion Group: {list(dataset.distortion_groups.keys())[label]}")
 """

# # 왜곡 분류 세분화(# class=25)
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path (str): CSV 파일 경로
            img_dir (str): 왜곡 이미지 폴더 경로
            transform (callable, optional): 이미지 변환 함수
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # ✅ ref 이미지는 제외: dist_img 컬럼에서 '_' 문자가 포함된 행만 유지
        self.data = self.data[self.data['dist_img'].str.contains("_")].reset_index(drop=True)

    def get_distortion_index(self, dist_img_name):
        """
        파일명에서 왜곡 코드(01~25)를 추출하고 0~24로 변환
        예: I01_14_03.png → '14' → 13
        """
        distortion_code = dist_img_name.split("_")[1]  # '14'
        return int(distortion_code) - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dist_img_name = row['dist_img']
        dist_img_path = os.path.join(self.img_dir, dist_img_name)

        # ✅ 이미지 로드
        image = Image.open(dist_img_path).convert("RGB")

        # ✅ 변환 적용
        if self.transform:
            image = self.transform(image)

        # ✅ 라벨 추출 (0~24)
        label = self.get_distortion_index(dist_img_name)

        return image, label


# ✅ 테스트 코드
if __name__ == "__main__":
    csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
    img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = KADID10KDataset(csv_path, img_dir, transform=transform)

    # ✅ 샘플 출력
    img, label = dataset[0]
    print(f"✅ 이미지 텐서 크기: {img.shape}")
    print(f"✅ 라벨 (0~24): {label}")
