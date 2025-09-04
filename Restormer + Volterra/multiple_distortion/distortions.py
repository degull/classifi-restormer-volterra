# distortions.py
import numpy as np
from PIL import Image, ImageFilter
import cv2

# ---------------- Blur ----------------
def gaussian_blur(img, level):
    return img.filter(ImageFilter.GaussianBlur(radius=level))

def motion_blur(img, level):
    arr = np.array(img)
    size = 3 + 2 * level  # 5~13
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0
    kernel /= size
    arr = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(arr)

# ---------------- Noise ----------------
def gaussian_noise(img, level):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, level * 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def impulse_noise(img, level):
    arr = np.array(img)
    prob = 0.01 * level
    noisy = arr.copy()
    mask = np.random.rand(*arr.shape[:2])
    noisy[mask < prob/2] = 0
    noisy[mask > 1 - prob/2] = 255
    return Image.fromarray(noisy)

# ---------------- Rain ----------------
def add_rain(img, level):
    arr = np.array(img).astype(np.uint8)
    h, w, _ = arr.shape
    rain = np.zeros((h, w), dtype=np.uint8)

    num_streaks = 300 * level
    for _ in range(num_streaks):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(5, 15)
        cv2.line(rain, (x1, y1), (x1, min(h-1, y1+length)), 255, 1)

    rain = cv2.blur(rain, (3, 3))
    rain = cv2.cvtColor(rain, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(arr, 1.0, rain, 0.2, 0)
    return Image.fromarray(blended)

# ---------------- Snow ----------------
def add_snow(img, level):
    arr = np.array(img).astype(np.uint8)
    h, w, _ = arr.shape
    snow = np.zeros((h, w), dtype=np.uint8)

    num_flakes = 200 * level
    for _ in range(num_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        cv2.circle(snow, (x, y), radius=np.random.randint(1, 3), color=255, thickness=-1)

    snow = cv2.GaussianBlur(snow, (3, 3), 0)
    snow = cv2.cvtColor(snow, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(arr, 1.0, snow, 0.3, 0)
    return Image.fromarray(blended)
