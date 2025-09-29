import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SIDD_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: E:/restormer+volterra/data/SIDD
        실제 이미지 경로: root_dir/Data/{scene}/GT_SRGB_010.PNG 등
        """
        self.data_dir = os.path.join(root_dir, "Data")
        self.pairs = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        for scene_dir in sorted(os.listdir(self.data_dir)):
            scene_path = os.path.join(self.data_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            noisy_path = os.path.join(scene_path, 'NOISY_SRGB_010.PNG')
            clean_path = os.path.join(scene_path, 'GT_SRGB_010.PNG')

            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                self.pairs.append((noisy_path, clean_path))
            else:
                print(f"[⚠️] 누락된 파일: {scene_path}")

        if not self.pairs:
            raise ValueError(f"[❌] 유효한 이미지 쌍을 찾을 수 없습니다: {self.data_dir}")
        else:
            print(f"[✅] 총 {len(self.pairs)} 쌍의 이미지가 로드되었습니다.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = Image.open(noisy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        return self.transform(noisy), self.transform(clean)


# ✅ 디버깅용 확인 코드
if __name__ == '__main__':
    root_path = 'E:/restormer+volterra/data/SIDD'
    dataset = SIDD_Dataset(root_dir=root_path)

    print(f"총 이미지 쌍 개수: {len(dataset)}")
    noisy, clean = dataset[0]
    print(f"Noisy 이미지 크기: {noisy.shape}")
    print(f"Clean 이미지 크기: {clean.shape}")
    print(f"Noisy tensor 값 범위: min={noisy.min().item()}, max={noisy.max().item()}")
    print(f"Clean tensor 값 범위: min={clean.min().item()}, max={clean.max().item()}")
