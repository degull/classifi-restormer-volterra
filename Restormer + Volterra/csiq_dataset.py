# E:/MRVNet2D/Restormer + Volterra/csiq_dataset.py

import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class CSIQDataset(Dataset):
    def __init__(self, csv_file, root_dir='', transform=None):
        """
        Args:
            csv_file (str): E:/MRVNet2D/dataset/CSIQ/CSIQ.txt 파일 경로
            root_dir (str): 이미지 경로 앞에 붙일 root 디렉토리
            transform (callable, optional): 이미지에 적용할 transform
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 상대 경로를 OS에 맞게 변환
        distorted_rel = self.data.iloc[idx]['dis_img_path'].replace('/', os.sep)
        reference_rel = self.data.iloc[idx]['ref_img_path'].replace('/', os.sep)

        # ✅ "CSIQ/" 제거 (중복 방지)
        if distorted_rel.startswith("CSIQ" + os.sep):
            distorted_rel = distorted_rel[len("CSIQ" + os.sep):]
        if reference_rel.startswith("CSIQ" + os.sep):
            reference_rel = reference_rel[len("CSIQ" + os.sep):]

        distorted_path = os.path.join(self.root_dir, distorted_rel)
        reference_path = os.path.join(self.root_dir, reference_rel)

        # 절대 경로가 아닌 경우 root_dir와 결합
        distorted_path = distorted_rel if os.path.isabs(distorted_rel) else os.path.normpath(os.path.join(self.root_dir, distorted_rel))
        reference_path = reference_rel if os.path.isabs(reference_rel) else os.path.normpath(os.path.join(self.root_dir, reference_rel))

        # 이미지 로딩
        distorted = Image.open(distorted_path).convert('RGB')
        reference = Image.open(reference_path).convert('RGB')

        # 전처리 적용
        if self.transform:
            distorted = self.transform(distorted)
            reference = self.transform(reference)

        return distorted, reference
