#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import numpy as np 
import cv2  

from .image_utils import ensure_gray, ensure_color, to_float, to_uint8, auto_noise_variance


def _colorfulness_bgr(img):
    img = ensure_color(img).astype(np.float32)
    (B, G, R) = cv2.split(img)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))
    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


def _edge_ratio(gray: np.ndarray) -> float:
    gray8 = gray if gray.dtype == np.uint8 else to_uint8(to_float(gray))
    v = float(np.median(gray8))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray8, lower, upper)
    return float(np.mean(edges > 0))


def compute_image_metrics(img) -> dict:
    gray = ensure_gray(img)
    gray_f = to_float(gray)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    noise_var = float(auto_noise_variance(gray_f, 3))
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    dyn_range = float((np.max(gray) - np.min(gray)) / 255.0)
    edges_ratio = _edge_ratio(gray)
    colorfulness = float(_colorfulness_bgr(img)) if img.ndim == 3 else 0.0
    gray8 = gray if gray.dtype == np.uint8 else to_uint8(gray_f)
    near_black = np.mean(gray8 <= 5)
    near_white = np.mean(gray8 >= 250)
    impulse_ratio = float(near_black + near_white)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": int(1 if img.ndim == 2 else img.shape[2]),
        "var_laplacian": var_lap,
        "noise_var": noise_var,
        "mean_brightness": mean_brightness,
        "std_contrast": std_contrast,
        "dynamic_range": dyn_range,
        "edges_ratio": edges_ratio,
        "colorfulness": colorfulness,
        "impulse_noise_ratio": impulse_ratio,
    }


__all__ = ["compute_image_metrics"]

