import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SIDD_Dataset(Dataset):
    def __init__(self, root_or_csv, transform=None):
        """
        Args:
            root_or_csv (str):
              - 폴더 기반: E:/restormer+volterra/data/SIDD
                (내부 구조: Data/{scene}/NOISY_SRGB_xxx.PNG, GT_SRGB_xxx.PNG)
              - CSV 기반: E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv
                (csv: noisy_path, clean_path)
        """
        self.pairs = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        if root_or_csv.lower().endswith(".csv"):
            # ✅ CSV 기반 로드
            df = pd.read_csv(root_or_csv)
            for _, row in df.iterrows():
                noisy_path, clean_path = row[0], row[1]
                if os.path.exists(noisy_path) and os.path.exists(clean_path):
                    self.pairs.append((noisy_path, clean_path))
                else:
                    print(f"[⚠️] 누락된 파일: {noisy_path}, {clean_path}")
        else:
            # ✅ 폴더 기반 로드
            self.data_dir = os.path.join(root_or_csv, "Data")
            for scene_dir in sorted(os.listdir(self.data_dir)):
                scene_path = os.path.join(self.data_dir, scene_dir)
                if not os.path.isdir(scene_path):
                    continue

                noisy_path = os.path.join(scene_path, "NOISY_SRGB_010.PNG")
                clean_path = os.path.join(scene_path, "GT_SRGB_010.PNG")

                if os.path.exists(noisy_path) and os.path.exists(clean_path):
                    self.pairs.append((noisy_path, clean_path))
                else:
                    print(f"[⚠️] 누락된 파일: {scene_path}")

        if not self.pairs:
            raise ValueError(f"[❌] 유효한 이미지 쌍을 찾을 수 없습니다: {root_or_csv}")
        else:
            print(f"[✅] 총 {len(self.pairs)} 쌍의 이미지가 로드되었습니다.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = Image.open(noisy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        return self.transform(noisy), self.transform(clean)


# ✅ 디버깅용 실행
if __name__ == "__main__":
    # CSV 기반
    csv_path = r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv"
    dataset_csv = SIDD_Dataset(csv_path)
    print(f"[CSV] 총 이미지 쌍 개수: {len(dataset_csv)}")

    # 폴더 기반
    root_path = r"E:/restormer+volterra/data/SIDD"
    dataset_folder = SIDD_Dataset(root_path)
    print(f"[FOLDER] 총 이미지 쌍 개수: {len(dataset_folder)}")

    noisy, clean = dataset_csv[0]
    print(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
