#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import time
from typing import Optional

import cv2  
import numpy as np 


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def ensure_color(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def auto_noise_variance(img_f: np.ndarray, ksize: int = 5) -> float:
    """Stima semplice del rumore: media della varianza locale."""
    mu = cv2.blur(img_f, (ksize, ksize))
    mu2 = cv2.blur(img_f * img_f, (ksize, ksize))
    var = np.maximum(mu2 - mu * mu, 0.0)
    return float(np.mean(var))


__all__ = [
    "to_float",
    "to_uint8",
    "ensure_color",
    "ensure_gray",
    "timestamp",
    "auto_noise_variance",
]

