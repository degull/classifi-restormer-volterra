import os, time, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# --------------------------------------------------------------
# ✅ 모델 import
# --------------------------------------------------------------
from models.restormer_volterra import RestormerVolterra

# --------------------------------------------------------------
# ✅ 데이터셋 경로 설정 (Rain100H 예시)
# --------------------------------------------------------------
GT_DIR = r"E:/restormer+volterra/data/rain100H/test/norain"
INPUT_DIR = r"E:/restormer+volterra/data/rain100H/test/rain"

device = "cuda" if torch.cuda.is_available() else "cpu"
to_tensor = transforms.ToTensor()


# --------------------------------------------------------------
# ✅ PSNR / SSIM 계산 함수
# --------------------------------------------------------------
def evaluate_folder(model, input_dir, gt_dir):
    psnr_total, ssim_total = 0, 0
    count = 0
    model.eval()
    with torch.no_grad():
        for fname in tqdm(os.listdir(input_dir), desc="Evaluating"):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            inp = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            gt = Image.open(os.path.join(gt_dir, fname)).convert("RGB")

            inp_t = to_tensor(inp).unsqueeze(0).to(device)
            gt_t = to_tensor(gt).unsqueeze(0).to(device)

            # forward
            with torch.cuda.amp.autocast():
                pred = model(inp_t)
            pred = torch.clamp(pred, 0, 1)

            # metric 계산
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_val = psnr_metric(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim_metric(gt_np, pred_np, data_range=1.0, channel_axis=2)

            psnr_total += psnr_val
            ssim_total += ssim_val
            count += 1

    return psnr_total / count, ssim_total / count


# --------------------------------------------------------------
# ✅ 모델 설정 조합
# --------------------------------------------------------------
configs = {
    "A_Baseline": dict(use_volterra_mdta=False, use_volterra_gdfn=False),
    "B_Vol_MDTA": dict(use_volterra_mdta=True, use_volterra_gdfn=False),
    "C_Vol_GDFN": dict(use_volterra_mdta=False, use_volterra_gdfn=True),
    "D_Full_VET": dict(use_volterra_mdta=True, use_volterra_gdfn=True),
}

results = {}

# --------------------------------------------------------------
# ✅ 실험 루프
# --------------------------------------------------------------
for name, cfg in configs.items():
    print(f"\n===== Testing {name} =====")
    model = RestormerVolterra(**cfg).to(device)
    model.eval()

    # warm-up
    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    # inference 시간 측정
    N = 30
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            _ = model(dummy)
    torch.cuda.synchronize()
    avg_time = (time.time() - t0) / N * 1000  # ms 단위

    # VRAM 사용량
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    torch.cuda.reset_peak_memory_stats()

    # 성능 평가 (Rain100H test)
    psnr, ssim = evaluate_folder(model, INPUT_DIR, GT_DIR)

    results[name] = dict(PSNR=psnr, SSIM=ssim, Time=avg_time, VRAM=mem)

# --------------------------------------------------------------
# ✅ 결과 출력
# --------------------------------------------------------------
print("\n===== Ablation Results (Rain100H) =====")
print(f"{'Model':<15} {'PSNR':<8} {'SSIM':<8} {'Time(ms)':<10} {'VRAM(MB)':<10}")
for k, v in results.items():
    print(f"{k:<15} {v['PSNR']:<8.3f} {v['SSIM']:<8.4f} {v['Time']:<10.2f} {v['VRAM']:<10.1f}")

