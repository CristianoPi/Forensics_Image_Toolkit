#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import os
import cv2  # type: ignore
import numpy as np  # type: ignore

from forensic_cli.core.image_utils import timestamp


def salva_output(img: np.ndarray, base_name: str, suffix: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    root, ext = os.path.splitext(os.path.basename(base_name))
    fname = f"{root}_{suffix}_{timestamp()}{ext if ext else '.png'}"
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, img)
    return out_path


__all__ = ["salva_output"]

