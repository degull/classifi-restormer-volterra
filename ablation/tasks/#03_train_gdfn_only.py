# E:/restormer+volterra/ablation/tasks/train_gdfn_only.py
import sys, os, time, random, torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
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

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-4

# ✅ 저장 경로 (G:\gdfn_only)
SAVE_DIR = r"G:\gdfn_only"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_DIR = r"E:/restormer+volterra/data/rain100H/train"
TEST_DIR  = r"E:/restormer+volterra/data/rain100H/test"

# ---------------- Evaluation ----------------
def evaluate(model, loader, max_eval=10):
    """평가 시 랜덤 10장만 사용 (속도 향상)"""
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for i, (inp, gt) in enumerate(loader):
            if i >= max_eval:
                break
            inp, gt = inp.to(DEVICE), gt.to(DEVICE)
            out = model(inp)
            out_np = out[0].detach().cpu().permute(1, 2, 0).numpy()
            gt_np  = gt[0].detach().cpu().permute(1, 2, 0).numpy()
            psnr_scores.append(compute_psnr(gt_np, out_np, data_range=1.0))
            ssim_scores.append(compute_ssim(gt_np, out_np, channel_axis=2, data_range=1.0))
    return float(sum(psnr_scores) / len(psnr_scores)), float(sum(ssim_scores) / len(ssim_scores))

# ---------------- Main ----------------
if __name__ == "__main__":
    # ---------------- Dataset ----------------
    full_train = Rain100HDataset(TRAIN_DIR)
    full_test  = Rain100HDataset(TEST_DIR)

    # ⚡ 빠른 테스트용: 일부 샘플만 사용
    train_indices = random.sample(range(len(full_train)), min(200, len(full_train)))
    test_indices  = random.sample(range(len(full_test)),  min(20, len(full_test)))
    train_dataset = Subset(full_train, train_indices)
    test_dataset  = Subset(full_test, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"[INFO] Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

    # ---------------- Model ----------------
    # ✅ Volterra: GDFN 모듈에만 적용
    model = RestormerVolterra(use_volterra_mdta=False, use_volterra_gdfn=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    # ---------------- Training ----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for inp, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            inp, gt = inp.to(DEVICE), gt.to(DEVICE)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        psnr, ssim = evaluate(model, test_loader, max_eval=10)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}: Loss={epoch_loss / len(train_loader):.4f}, "
              f"PSNR={psnr:.2f}, SSIM={ssim:.4f}, Time={elapsed:.1f}s")

        # ✅ safe save (Windows zip writer 버그 회피)
        save_path = os.path.join(SAVE_DIR, f"gdfn_only_epoch{epoch}_psnr{psnr:.2f}_ssim{ssim:.4f}.pth")
        torch.save(
            model.state_dict(),
            save_path,
            _use_new_zipfile_serialization=False
        )
