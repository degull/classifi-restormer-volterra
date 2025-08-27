# pipeline.py
# 데이터셋 그대로 쓰는 generator (dataset-driven)
""" import random, numpy as np
from distortions import *

# Restoration task-specific groups
DISTORTION_GROUPS = {
    "rain": [add_rain],
    "snow": [add_snow],
    "blur": [gaussian_blur, motion_blur],
}

def apply_random_distortions(img, Ndist=4, return_info=False):
    # 몇 개 distortion 적용할지 (1 ~ Ndist 랜덤)
    ndist = random.randint(1, Ndist)
    chosen_groups = random.sample(list(DISTORTION_GROUPS.keys()), ndist)
    chosen = [random.choice(DISTORTION_GROUPS[g]) for g in chosen_groups]
    random.shuffle(chosen)

    out = img.copy()
    info = []
    for fn in chosen:
        # 강도: Gaussian 분포 샘플링 (1~5)
        level = int(np.clip(np.random.normal(2.5, 1.0), 1, 5))
        out = fn(out, level)
        info.append(f"{fn.__name__}(L{level})")

    if return_info:
        return out, info
    return out
 """

# pipeline.py
import os
import glob
from PIL import Image

# ---------------- Utils ----------------
def load_img(path):
    return Image.open(path).convert("RGB")

# ---------------- Rain100H / Rain100L ----------------
def sample_rain100_pair(base_dir, fname):
    """
    Rain100H/L: base_dir 안에 'rain', 'norain' 폴더가 있음
    fname: 이미지 파일 이름
    return: (rain_img, gt_img)
    """
    rain_path = os.path.join(base_dir, "rain", fname)
    gt_path   = os.path.join(base_dir, "norain", fname)
    return load_img(rain_path), load_img(gt_path)

# ---------------- CSD (Desnow) ----------------
def sample_csd_pair(base_dir, fname):
    """
    CSD: base_dir 안에 'Snow', 'Gt' 폴더가 있음
    fname: 이미지 파일 이름
    return: (snow_img, gt_img)
    """
    snow_path = os.path.join(base_dir, "Snow", fname)
    gt_path   = os.path.join(base_dir, "Gt", fname)
    return load_img(snow_path), load_img(gt_path)

# ---------------- HIDE (Deblur) ----------------
def sample_hide_pair(blur_path, gt_dir):
    """
    HIDE: blurred image와 GT image가 같은 파일명으로 존재
    blur_path: 흐린 이미지 경로
    gt_dir: GT 폴더 경로
    return: (blurred_img, gt_img)
    """
    fname = os.path.basename(blur_path)
    gt_path = os.path.join(gt_dir, fname)
    return load_img(blur_path), load_img(gt_path)

# ---------------- SIDD (Denoise) ----------------
def sample_sidd_pair(folder):
    """
    SIDD: 각 시퀀스 폴더에 NOISY_SRGB_xxx.PNG 와 GT_SRGB_xxx.PNG 가 쌍으로 존재
    folder: 시퀀스 폴더 경로
    return: (noisy_img, gt_img)
    """
    noisy_files = glob.glob(os.path.join(folder, "NOISY_SRGB_*.PNG"))
    gt_files    = glob.glob(os.path.join(folder, "GT_SRGB_*.PNG"))
    noisy_files.sort()
    gt_files.sort()
    if len(noisy_files) == 0 or len(gt_files) == 0:
        raise FileNotFoundError(f"No noisy/GT pair found in {folder}")
    return load_img(noisy_files[0]), load_img(gt_files[0])

# ---------------- Main Interface ----------------
def dataset_driven_generator(dataset_name, path_info):
    """
    dataset_name: 'rain100H', 'rain100L', 'CSD', 'HIDE', 'SIDD'
    path_info: dict with dataset-specific info
        - Rain100H/L: {"base_dir": "path/to/rain100H/train", "fname": "xxx.png"}
        - CSD: {"base_dir": "path/to/CSD/Test", "fname": "xxx.tif"}
        - HIDE: {"blur_path": "path/to/HIDE/test/xxx.png", "gt_dir": "path/to/HIDE/GT"}
        - SIDD: {"folder": "path/to/SIDD/Data/0001_001_xxx"}
    return: (input_img, gt_img)
    """
    if dataset_name in ["rain100H", "rain100L"]:
        return sample_rain100_pair(path_info["base_dir"], path_info["fname"])
    elif dataset_name == "CSD":
        return sample_csd_pair(path_info["base_dir"], path_info["fname"])
    elif dataset_name == "HIDE":
        return sample_hide_pair(path_info["blur_path"], path_info["gt_dir"])
    elif dataset_name == "SIDD":
        return sample_sidd_pair(path_info["folder"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
