import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SIDDDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): SIDD 데이터셋 루트 디렉토리 경로
            split (str): 'train' 또는 'test'
        """
        self.split = split
        self.noisy_dir = os.path.join(root_dir, split, 'noisy')
        self.clean_dir = os.path.join(root_dir, split, 'clean')
        self.image_names = sorted(os.listdir(self.noisy_dir))
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.image_names[idx])
        clean_path = os.path.join(self.clean_dir, self.image_names[idx])

        noisy = Image.open(noisy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        noisy = self.transform(noisy)
        clean = self.transform(clean)

        return noisy, clean
