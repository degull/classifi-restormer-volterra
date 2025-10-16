# E:/restormer+volterra/ablation/tasks/test_gdfn_only_all.py
import sys, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ---------------- Path 설정 ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "re_dataset"))

# ---------------- Import ----------------
from models.restormer_volterra import RestormerVolterra
from re_dataset.rain100h_dataset import Rain100HDataset
from re_dataset.rain100l_dataset import Rain100LDataset
from re_dataset.gopro_dataset import GoProDataset
from re_dataset.sidd_dataset import SIDD_Dataset
from re_dataset.csd_dataset import CSDDataset

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"E:\restormer+volterra\ablation\checkpoints\train_gdfn_only\gdfn_only_epoch12_psnr24.33_ssim0.7598.pth"
BATCH_SIZE = 1

# ---------------- Evaluation ----------------
def evaluate(model, loader, max_eval=None):
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for i, (inp, gt) in enumerate(tqdm(loader, leave=False)):
            if max_eval and i >= max_eval:  # ⚡ 빠른 테스트
                break
            inp, gt = inp.to(DEVICE), gt.to(DEVICE)
            out = model(inp)

            out_np = out[0].detach().cpu().permute(1, 2, 0).numpy()
            gt_np  = gt[0].detach().cpu().permute(1, 2, 0).numpy()

            psnr_scores.append(compute_psnr(gt_np, out_np, data_range=1.0))
            ssim_scores.append(compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0))

    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))

# ---------------- Main ----------------
if __name__ == "__main__":
    # ✅ 모델 로드 (GDFN only)
    model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=True).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"[INFO] Loaded checkpoint: {CKPT}")

    # ✅ 데이터셋 로드
    DATASETS = {
        "Rain100H": Rain100HDataset(r"E:/restormer+volterra/data/rain100H/test"),
        "Rain100L": Rain100LDataset(r"E:/restormer+volterra/data/rain100L/test"),
        "GoPro":    GoProDataset(r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv"),
        "SIDD":     SIDD_Dataset(r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv"),
        "CSD":      CSDDataset(r"E:/restormer+volterra/data/CSD/Test"),
    }

    results = {}
    for name, dataset in DATASETS.items():
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"\n[INFO] Evaluating {name} | {len(dataset)} samples")
        psnr, ssim = evaluate(model, loader, max_eval=50)  # ⚡ 50장만 평가
        results[name] = (psnr, ssim)
        print(f"[RESULT] {name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    # ✅ 최종 요약
    print("\n==== Final Results (GDFN Only + Volterra) ====")
    for name, (psnr, ssim) in results.items():
        print(f"{name:10s} | PSNR={psnr:.2f} | SSIM={ssim:.4f}")
