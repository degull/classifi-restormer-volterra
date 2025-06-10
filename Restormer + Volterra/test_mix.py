# E:/MRVNet2D/Restormer + Volterra/test.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from restormer_volterra import RestormerVolterra
from kadid_dataset import KADID10KDataset
from tid_dataset import TID2013Dataset
from csiq_dataset import CSIQDataset

from PIL import Image

# ✅ 경로 설정
MODEL_PATH = 'checkpoints/restormer_volterra_all/epoch_100.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KADID_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
TID_CSV = 'E:/MRVNet2D/dataset/tid2013/mos.csv'
TID_DISTORTED_DIR = 'E:/MRVNet2D/dataset/tid2013/distorted_images'
TID_REFERENCE_DIR = 'E:/MRVNet2D/dataset/tid2013/reference_images'
CSIQ_CSV = 'E:/MRVNet2D/dataset/CSIQ/CSIQ.txt'
CSIQ_ROOT = 'E:/MRVNet2D/dataset/CSIQ'

BATCH_SIZE = 1  # 평가에서는 1장씩

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@torch.no_grad()
def evaluate(model, dataloader, name):
    model.eval()
    psnr_list, ssim_list = [], []

    for distorted, reference in tqdm(dataloader, desc=f"Testing on {name}"):
        distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
        output = model(distorted)

        # Tensor → Numpy
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reference_np = reference.squeeze(0).permute(1, 2, 0).cpu().numpy()

        output_np = np.clip(output_np, 0, 1)

        psnr = compare_psnr(reference_np, output_np, data_range=1.0)
        ssim = compare_ssim(reference_np, output_np, multichannel=True, data_range=1.0)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(f"[{name}] PSNR: {np.mean(psnr_list):.4f}, SSIM: {np.mean(ssim_list):.4f}")

def main():
    model = RestormerVolterra().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # ✅ 테스트 데이터셋 정의
    kadid_dataset = KADID10KDataset(csv_file=KADID_CSV, transform=transform)
    tid_dataset = TID2013Dataset(csv_file=TID_CSV, distorted_dir=TID_DISTORTED_DIR, reference_dir=TID_REFERENCE_DIR, transform=transform)
    csiq_dataset = CSIQDataset(csv_file=CSIQ_CSV, root_dir=CSIQ_ROOT, transform=transform)

    kadid_loader = DataLoader(kadid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    tid_loader = DataLoader(tid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    csiq_loader = DataLoader(csiq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ✅ 평가
    evaluate(model, kadid_loader, "KADID10K")
    evaluate(model, tid_loader, "TID2013")
    evaluate(model, csiq_loader, "CSIQ")

if __name__ == "__main__":
    main()
