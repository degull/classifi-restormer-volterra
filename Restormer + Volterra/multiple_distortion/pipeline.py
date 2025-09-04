# pipeline.py
import random, numpy as np
from distortions import *

# Restoration-related groups only
DISTORTION_GROUPS = {
    "rain": [add_rain],
    "snow": [add_snow],
    "blur": [gaussian_blur, motion_blur],
    "noise": [gaussian_noise, impulse_noise],
}

def apply_random_distortions(img, Ndist=4, return_info=False):
    # 몇 개 distortion 적용할지 (1 ~ min(Ndist, 그룹 수))
    ndist = random.randint(1, min(Ndist, len(DISTORTION_GROUPS)))
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
