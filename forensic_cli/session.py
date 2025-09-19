#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from forensic_cli.core.image_utils import ensure_color


@dataclass
class Session:
    img_path: Optional[str] = None
    original: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    out_dir: str = os.path.join(os.getcwd(), "outputs")

    def load_image(self, path: str) -> bool:
        if not os.path.exists(path):
            print("[ERRORE] Il path specificato non esiste.")
            return False
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("[ERRORE] Impossibile aprire l'immagine. Formato non supportato o file corrotto.")
            return False
        if img.ndim == 2:
            pass
        elif img.ndim == 3 and img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = bgr
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            img = cv2.convertScaleAbs(img)
        self.img_path = path
        self.original = img.copy()
        self.current = img.copy()
        print(f"[OK] Immagine caricata: {img.shape[1]}x{img.shape[0]} px, {1 if img.ndim==2 else img.shape[2]} canali, dtype={img.dtype}")
        return True

    def require_image(self) -> bool:
        if self.current is None:
            print("[INFO] Nessuna immagine caricata.")
            return False
        return True


__all__ = ["Session"]

