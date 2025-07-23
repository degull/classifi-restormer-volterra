import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

from restormer_volterra import RestormerVolterra

# ----------------------- Config -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"E:\restormer+volterra\checkpoints\jpeg_bsds500\epoch_86_ssim0.9994_psnr50.24.pth"
QUALITY_LEVELS = [10, 20, 30]
DATA_ROOT = r"E:/restormer+volterra/data"
SAVE_ROOT = r"E:/restormer+volterra/results"

DATASETS = {
    "classic5": [
        {
            "quality": qf,
            "img_dir": os.path.join(DATA_ROOT, f"classic5/gray/qf_{qf}"),
            "gt_dir": os.path.join(DATA_ROOT, "classic5/refimgs"),
            "save_dir": os.path.join(SAVE_ROOT, f"classic5/qf_{qf}")
        } for qf in QUALITY_LEVELS
    ],
    "live1": [
        {
            "quality": qf,
            "img_dir": os.path.join(DATA_ROOT, f"live1/color/qf_{qf}"),
            "gt_dir": os.path.join(DATA_ROOT, "live1/refimgs"),
            "save_dir": os.path.join(SAVE_ROOT, f"live1/qf_{qf}")
        } for qf in QUALITY_LEVELS
    ],
    "bsd500": [
        {
            "quality": qf,
            "img_dir": os.path.join(DATA_ROOT, f"BSD500/color/qf_{qf}"),
            "gt_dir": os.path.join(DATA_ROOT, "BSD500/refimgs"),
            "save_dir": os.path.join(SAVE_ROOT, f"BSD500/qf_{qf}")
        } for qf in QUALITY_LEVELS
    ]
}

# ----------------------- Dataset -----------------------
class TestJPEGDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        if "classic5" in gt_dir.lower() or "live1" in gt_dir.lower():
            ext = ".bmp"
        else:
            ext = ".jpg"
        self.gt_paths = [
            os.path.join(gt_dir, os.path.splitext(os.path.basename(f))[0] + ext)
            for f in self.img_paths
        ]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        gt = Image.open(self.gt_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)
        gt_tensor = self.transform(gt)
        return img_tensor, gt_tensor, os.path.basename(self.img_paths[idx]), img, gt

# ----------------------- Visualization -----------------------
def save_comparison_image(input_img, gt_img, restored_img, save_path, psnr_input, ssim_input, psnr_restored, ssim_restored):
    input_img = input_img.resize((256, 256))
    gt_img = gt_img.resize((256, 256))
    restored_img = restored_img.resize((256, 256))

    combined = Image.new("RGB", (256 * 3, 256))
    combined.paste(input_img, (0, 0))
    combined.paste(gt_img, (256, 0))
    combined.paste(restored_img, (512, 0))

    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw.text((10, 5), f"Input\nPSNR:{psnr_input:.2f} SSIM:{ssim_input:.4f}", font=font, fill=(255, 0, 0))
    draw.text((266, 5), f"GT", font=font, fill=(0, 255, 0))
    draw.text((522, 5), f"Output\nPSNR:{psnr_restored:.2f} SSIM:{ssim_restored:.4f}", font=font, fill=(0, 0, 255))

    combined.save(save_path)

# ----------------------- Test One Dataset -----------------------
def test_one_dataset(dataset_name, config, model):
    os.makedirs(config["save_dir"], exist_ok=True)
    dataset = TestJPEGDataset(config["img_dir"], config["gt_dir"])

    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for img_tensor, gt_tensor, fname, input_pil, gt_pil in tqdm(dataset, desc=f"{dataset_name.upper()} QF={config['quality']}"):
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            gt_tensor = gt_tensor.unsqueeze(0).to(DEVICE)
            restored = model(img_tensor).clamp(0, 1)

            pred_np = restored[0].cpu().numpy().transpose(1, 2, 0)
            gt_np = gt_tensor[0].cpu().numpy().transpose(1, 2, 0)
            input_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)

            psnr_restored = compute_psnr(gt_np, pred_np, data_range=1.0)
            ssim_restored = compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=-1)
            psnr_input = compute_psnr(gt_np, input_np, data_range=1.0)
            ssim_input = compute_ssim(gt_np, input_np, data_range=1.0, channel_axis=-1)

            total_psnr += psnr_restored
            total_ssim += ssim_restored

            restored_img = Image.fromarray((pred_np * 255).astype(np.uint8))
            save_path = os.path.join(config["save_dir"], fname)
            save_comparison_image(input_pil, gt_pil, restored_img, save_path, psnr_input, ssim_input, psnr_restored, ssim_restored)

    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    return avg_psnr, avg_ssim

# ----------------------- Main -----------------------
def main():
    model = RestormerVolterra().to(DEVICE)
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    print(f"{'Dataset':<10} | {'Quality':<7} | {'PSNR':<6} | {'SSIM':<6}")
    print("-" * 40)

    for dataset_name, configs in DATASETS.items():
        for config in configs:
            psnr, ssim = test_one_dataset(dataset_name, config, model)
            print(f"{dataset_name:<10} | {config['quality']:<7} | {psnr:.2f} | {ssim:.4f}")

if __name__ == "__main__":
    main()
