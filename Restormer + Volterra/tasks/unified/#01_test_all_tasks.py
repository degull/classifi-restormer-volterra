import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra


class PairedFolderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        self.input_paths = sorted(glob(os.path.join(input_dir, "**", "*.*"), recursive=True))
        self.input_paths = [f for f in self.input_paths if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]

        # GT를 확장자 제거 후 이름만 매칭되도록 수정
        gt_files = {os.path.splitext(f)[0]: os.path.join(target_dir, f)
                    for f in os.listdir(target_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))}

        self.target_paths = []
        for inp_path in self.input_paths:
            base = os.path.splitext(os.path.basename(inp_path))[0]
            if base in gt_files:
                self.target_paths.append(gt_files[base])
            else:
                raise FileNotFoundError(f"GT 파일이 없음 (입력: {inp_path}, base: {base})")

        self.transform = transform

        assert len(self.input_paths) == len(self.target_paths), \
            f"Mismatched input and GT lengths: {len(self.input_paths)} vs {len(self.target_paths)}"

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)


class PairedCSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)

        # ✅ 열 이름이 'dist_img', 'ref_img'인 경우 처리
        if 'dist_img' in df.columns and 'ref_img' in df.columns:
            self.paths = df[['dist_img', 'ref_img']].values.tolist()
        else:
            self.paths = df.values.tolist()

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.paths[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)



def evaluate(model, dataloader, name):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"[{name}]"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs).clamp(0, 1)

            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            for out, gt in zip(outputs, targets):
                out = np.transpose(out, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                psnr_total += compute_psnr(gt, out, data_range=1.0)
                ssim_total += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

    print(f"📌 {name:12s} → PSNR: {psnr_total/count:.2f} dB, SSIM: {ssim_total/count:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RestormerVolterra().to(device)

    checkpoint = r"E:\restormer+volterra\checkpoints\#01_all_tasks_balanced\epoch_96_ssim0.9309_psnr35.19.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"\n✅ Loaded checkpoint from {checkpoint}\n")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_sets = {
        "Rain100H": PairedFolderDataset(
            r"E:/restormer+volterra/data/rain100H/test/rain",
            r"E:/restormer+volterra/data/rain100H/test/norain", transform),
        "Rain100L": PairedFolderDataset(
            r"E:/restormer+volterra/data/rain100L/test/rain",
            r"E:/restormer+volterra/data/rain100L/test/norain", transform),
        "HIDE": PairedFolderDataset(
            r"E:/restormer+volterra/data/HIDE/test",
            r"E:/restormer+volterra/data/HIDE/GT", transform),
        "SIDD": PairedCSVDataset(
            r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv", transform),
        "CSD": PairedFolderDataset(
            r"E:/restormer+volterra/data/CSD/Test/Snow",
            r"E:/restormer+volterra/data/CSD/Test/Gt", transform),
        "BSD500": PairedFolderDataset(
            r"E:/restormer+volterra/data/BSD500/gray/qf_10",
            r"E:/restormer+volterra/data/BSD500/refimgs", transform),
        "Classic5": PairedFolderDataset(
            r"E:/restormer+volterra/data/classic5/gray/qf_10",
            r"E:/restormer+volterra/data/classic5/refimgs", transform),
        "LIVE1": PairedFolderDataset(
            r"E:/restormer+volterra/data/live1/gray/qf_10",
            r"E:/restormer+volterra/data/live1/refimgs", transform),
    }

    for name, dataset in test_sets.items():
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        evaluate(model, loader, name)
