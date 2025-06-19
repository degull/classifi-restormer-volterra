import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HIDEDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): e.g., "E:/restormer+volterra/data/HIDE"
            split (str): 'train' or 'test'
        """
        self.split = split
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        if split == 'train':
            self.input_dir = os.path.join(root_dir, 'train')
            self.gt_dir = os.path.join(root_dir, 'GT')
            self.image_names = sorted(os.listdir(self.input_dir))
        elif split == 'test':
            self.input_dir = os.path.join(root_dir, 'test')
            self.gt_dir = os.path.join(root_dir, 'GT')
            self.image_names = self._gather_test_images()
        else:
            raise ValueError("split must be 'train' or 'test'")

    def _gather_test_images(self):
        test_dirs = ['test-close-ups', 'test-long-shot']
        all_images = []
        for subdir in test_dirs:
            full_path = os.path.join(self.input_dir, subdir)
            all_images += [os.path.join(subdir, fname) for fname in os.listdir(full_path) if fname.endswith(('.jpg', '.png'))]
        return sorted(all_images)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        blur_name = self.image_names[idx]

        if self.split == 'train':
            blur_path = os.path.join(self.input_dir, blur_name)
            sharp_path = os.path.join(self.gt_dir, blur_name)
        else:  # test
            blur_path = os.path.join(self.input_dir, blur_name)
            sharp_path = os.path.join(self.gt_dir, os.path.basename(blur_name))

        # 🔍 디버깅용 경로 출력
        print(f"[{idx}] blur_path: {blur_path}")
        print(f"[{idx}] sharp_path: {sharp_path}")

        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')

        # 🔍 원본 이미지 크기 출력
        print(f"[{idx}] blur size: {blur.size}, sharp size: {sharp.size}")

        blur = self.transform(blur)
        sharp = self.transform(sharp)

        return blur, sharp
