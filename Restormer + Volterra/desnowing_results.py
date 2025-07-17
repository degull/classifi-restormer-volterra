import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from restormer_volterra import RestormerVolterra


class CSDDataset(Dataset):
    def __init__(self, snow_dir, gt_dir, transform=None):
        self.snow_files = sorted(os.listdir(snow_dir))
        self.snow_dir = snow_dir
        self.gt_dir = gt_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.snow_files)

    def __getitem__(self, idx):
        snow_path = os.path.join(self.snow_dir, self.snow_files[idx])
        gt_path = os.path.join(self.gt_dir, self.snow_files[idx])
        snow = Image.open(snow_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        return self.transform(snow), self.transform(gt)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SNOW_DIR = r"E:/restormer+volterra/data/CSD/Test/Snow"
TEST_GT_DIR = r"E:/restormer+volterra/data/CSD/Test/Gt"
RESULT_DIR = r"E:/restormer+volterra/results/desnowing_test"
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = r"E:/restormer+volterra/checkpoints/restormer_volterra_csd/epoch_27.pth"


def evaluate_and_save(model, dataloader):
    model.eval()
    total_psnr_out = total_ssim_out = 0.0
    total_psnr_in = total_ssim_in = 0.0
    count = 0

    with torch.no_grad():
        for idx, (snow, gt) in enumerate(dataloader):
            snow = snow.to(DEVICE)
            gt = gt.to(DEVICE)

            output = model(snow)

            gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
            snow_np = snow[0].cpu().numpy().transpose(1, 2, 0)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)

            # PSNR / SSIM 계산
            psnr_out = compute_psnr(gt_np, out_np, data_range=1.0)
            ssim_out = compute_ssim(gt_np, out_np, data_range=1.0, channel_axis=2, win_size=7)

            psnr_in = compute_psnr(gt_np, snow_np, data_range=1.0)
            ssim_in = compute_ssim(gt_np, snow_np, data_range=1.0, channel_axis=2, win_size=7)

            total_psnr_out += psnr_out
            total_ssim_out += ssim_out
            total_psnr_in += psnr_in
            total_ssim_in += ssim_in
            count += 1

            # 복원, 스노우, GT → PIL 변환
            out_img = (out_np * 255.0).clip(0, 255).astype(np.uint8)
            snow_img = (snow_np * 255.0).clip(0, 255).astype(np.uint8)
            gt_img = (gt_np * 255.0).clip(0, 255).astype(np.uint8)

            out_pil = Image.fromarray(out_img)
            snow_pil = Image.fromarray(snow_img)
            gt_pil = Image.fromarray(gt_img)

            # 3장 가로로 붙이기
            w, h = gt_pil.size
            combined = Image.new('RGB', (w * 3, h + 50), (255, 255, 255))  # 상단 여백 추가
            combined.paste(gt_pil, (0, 50))
            combined.paste(out_pil, (w, 50))
            combined.paste(snow_pil, (w * 2, 50))

            # 글씨 추가
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            title = "GT | Output | Snow"
            text1 = f"GT vs Output   PSNR: {psnr_out:.2f}dB  SSIM: {ssim_out:.4f}"
            text2 = f"GT vs Snow     PSNR: {psnr_in:.2f}dB  SSIM: {ssim_in:.4f}"

            draw.text((10, 5), title, font=font, fill=(0, 0, 0))
            draw.text((10, h + 5), text1, font=font, fill=(255, 0, 0))
            draw.text((10, h + 25), text2, font=font, fill=(255, 0, 0))

            combined.save(os.path.join(RESULT_DIR, f"{idx+1:03d}.png"))

            print(f"[{idx+1:02d}] GT vs Output  PSNR: {psnr_out:.2f} | SSIM: {ssim_out:.4f}")
            print(f"[{idx+1:02d}] GT vs Snow    PSNR: {psnr_in:.2f} | SSIM: {ssim_in:.4f} - Saved")

    avg_psnr_out = total_psnr_out / count
    avg_ssim_out = total_ssim_out / count
    avg_psnr_in = total_psnr_in / count
    avg_ssim_in = total_ssim_in / count
    print(f"\n✅ Average GT vs Output  PSNR: {avg_psnr_out:.2f} | SSIM: {avg_ssim_out:.4f}")
    print(f"✅ Average GT vs Snow    PSNR: {avg_psnr_in:.2f} | SSIM: {avg_ssim_in:.4f}")


def main():
    model = RestormerVolterra().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint)
    model.eval()

    test_ds = CSDDataset(TEST_SNOW_DIR, TEST_GT_DIR)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    evaluate_and_save(model, test_dl)


if __name__ == "__main__":
    main()
