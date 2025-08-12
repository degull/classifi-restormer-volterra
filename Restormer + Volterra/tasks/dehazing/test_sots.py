# E:/restormer+volterra/Restormer + Volterra/tasks/dehazing/test_sots.py
import os, sys, csv
from typing import List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
# models 폴더 접근 (RestormerVolterra import용)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from models.restormer_volterra import RestormerVolterra

# ---------------- Paths ----------------
BASE_DIR   = r"E:/restormer+volterra/data/SOTS"
SPLIT_DIR  = os.path.join(BASE_DIR, "splits")
RESULT_DIR = r"E:/restormer+volterra/results/sots"   # ✅ 복원 이미지 저장 루트
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 여기를 네 체크포인트 경로로 변경
CKPT = r"E:\restormer+volterra\checkpoints\sots_volterra\epoch_1_valssim0.8066_valpsnr20.16.pth"

os.makedirs(RESULT_DIR, exist_ok=True)

# ------------- Utils -------------
def read_pairs_csv(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # dist_img / ref_img 이름 가변 대응
        a = next((c for c in r.fieldnames if c.lower().startswith("dist")), None)
        b = next((c for c in r.fieldnames if c.lower().startswith("ref")),  None)
        if a is None or b is None:
            raise ValueError(f"CSV columns not found in {path}: {r.fieldnames}")
        for row in r:
            rows.append((row[a], row[b]))
    return rows

def _to_numpy01(t: torch.Tensor) -> np.ndarray:
    """ (1,C,H,W) or (C,H,W) tensor in [0,1] -> (H,W,3) float in [0,1] """
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)

def _save_img01(path: str, arr01: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (arr01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path)

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], size: int = 256, return_paths: bool = False):
        self.pairs = pairs
        self.return_paths = return_paths
        self.tfm = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x = Image.open(a).convert("RGB")
        y = Image.open(b).convert("RGB")
        if self.return_paths:
            return self.tfm(x), self.tfm(y), a
        return self.tfm(x), self.tfm(y)

@torch.no_grad()
def evaluate_and_save(model, dataloader, tag: str, save_root: str):
    """
    평가 + 복원 이미지 저장
    save_root/tag/파일명.png 로 저장됨
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for batch in tqdm(dataloader, desc=f"[Eval] {tag}"):
        # (x,y,path) 또는 (x,y)
        if len(batch) == 3:
            x, y, paths = batch
        else:
            x, y = batch
            paths = None

        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x).clamp(0, 1)

        # 저장
        for i in range(out.size(0)):
            out_np = _to_numpy01(out[i])
            if paths is not None:
                stem = os.path.splitext(os.path.basename(paths[i]))[0]
            else:
                stem = f"{tag}_{n+i:05d}"
            save_path = os.path.join(save_root, tag, f"{stem}.png")
            _save_img01(save_path, out_np)

        # 메트릭
        out_np = out.cpu().numpy()
        gt_np  = y.cpu().numpy()
        b = out_np.shape[0]
        for i in range(b):
            o = np.transpose(out_np[i], (1, 2, 0))
            g = np.transpose(gt_np[i],  (1, 2, 0))
            total_psnr += compute_psnr(g, o, data_range=1.0)
            total_ssim += compute_ssim(g, o, channel_axis=2, data_range=1.0)
            n += 1

    avg_psnr = (total_psnr / n) if n else 0.0
    avg_ssim = (total_ssim / n) if n else 0.0
    return avg_psnr, avg_ssim, n

def main():
    # 1) 스플릿 로드
    indoor_csv  = os.path.join(SPLIT_DIR, "indoor_test.csv")
    outdoor_csv = os.path.join(SPLIT_DIR, "outdoor_test.csv")

    if not (os.path.exists(indoor_csv) and os.path.exists(outdoor_csv)):
        raise FileNotFoundError("splits CSV가 없습니다. 먼저 make_sots_splits.py를 실행해 생성하세요.")

    ind_pairs = read_pairs_csv(indoor_csv)
    out_pairs = read_pairs_csv(outdoor_csv)

    # 저장 루트 (한 번 더 보장)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 저장 편의를 위해 path 반환하도록 구성
    ind_dl = DataLoader(PairDataset(ind_pairs, size=256, return_paths=True),
                        batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    out_dl = DataLoader(PairDataset(out_pairs,  size=256, return_paths=True),
                        batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 2) 모델 로드
    model = RestormerVolterra().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # DataParallel 호환
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✓ Loaded: {CKPT} | missing={len(missing)} unexpected={len(unexpected)}")

    # 3) 평가 + 저장
    ip, is_, n1 = evaluate_and_save(model, ind_dl, "indoor",  RESULT_DIR)
    op, os_, n2 = evaluate_and_save(model, out_dl, "outdoor", RESULT_DIR)

    tot_n = n1 + n2
    tot_p = (ip * n1 + op * n2) / tot_n if tot_n else 0.0
    tot_s = (is_ * n1 + os_ * n2) / tot_n if tot_n else 0.0

    print("\n=========== SOTS Test Summary ===========")
    print(f"Indoor  | images: {n1:4d} | PSNR: {ip:.2f} dB | SSIM: {is_:.4f}")
    print(f"Outdoor | images: {n2:4d} | PSNR: {op:.2f} dB | SSIM: {os_:.4f}")
    print("-----------------------------------------")
    print(f"Overall | images: {tot_n:4d} | PSNR: {tot_p:.2f} dB | SSIM: {tot_s:.4f}")
    print(f"🖼  Restored images saved to: {RESULT_DIR}\\{{indoor|outdoor}}")
    print("=========================================")

if __name__ == "__main__":
    main()
