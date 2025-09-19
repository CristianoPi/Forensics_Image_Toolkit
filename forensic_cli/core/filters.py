#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025

# Filtro e funzioni di elaborazione immagini per forensics
# Tutte le funzioni accettano immagini numpy e restituiscono immagini numpy
# Dipendenze: OpenCV (cv2), numpy
# Funzioni di utilità importate da image_utils (conversioni, normalizzazioni, ecc.)
from __future__ import annotations
from typing import Optional
import cv2 
import numpy as np 
from .image_utils import ensure_color, ensure_gray, to_float, to_uint8, auto_noise_variance



# Filtro mediano: riduce il rumore impulsivo (sale e pepe) preservando i bordi.
#   Sostituisce ogni pixel con la mediana dei valori nel suo intorno, efficace contro rumore impulsivo.
# Applica medianBlur su ogni canale se l'immagine è a colori.
def filtro_mediano(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = int(max(3, ksize) // 2 * 2 + 1)  # forza dispari >=3
    if img.ndim == 2:
        return cv2.medianBlur(img, ksize)
    # Per immagini a colori: applica il filtro separatamente su ogni canale
    channels = cv2.split(img)
    den = [cv2.medianBlur(c, ksize) for c in channels]
    return cv2.merge(den)



# Filtro gaussiano: riduce il rumore e ammorbidisce l'immagine tramite convoluzione con kernel gaussiano.
#   Applica una media pesata secondo una distribuzione gaussiana, attenua rumore e dettagli fini.
# ksize forza dispari >=3, sigma controlla la deviazione standard della gaussiana.
def filtro_gaussiano(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    ksize = int(max(3, ksize) // 2 * 2 + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)



# Filtro bilaterale: riduzione rumore che preserva i bordi.
#   Media pesata che tiene conto sia della distanza spaziale che della differenza di colore, preservando i bordi.
# d: diametro area pixel, sigma_color: filtro su differenza colore, sigma_space: filtro su distanza spaziale.
def filtro_bilaterale(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    d = int(max(1, d))
    return cv2.bilateralFilter(img, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))



# Filtro di Wiener adattivo: riduzione rumore gaussiano, stima locale della varianza.
#   Filtraggio adattivo che riduce il rumore stimando la varianza locale e quella del rumore, ottimo per rumore gaussiano.
# ksize: dimensione finestra locale, noise_var: varianza rumore (se None stimata automaticamente).
def filtro_wiener(img: np.ndarray, ksize: int = 5, noise_var: Optional[float] = None) -> np.ndarray:
    img_c = ensure_color(img)
    img_f = to_float(img_c)
    res = []
    for ch in cv2.split(img_f):
        # Media locale
        mu = cv2.blur(ch, (ksize, ksize))
        mu2 = cv2.blur(ch * ch, (ksize, ksize))
        var = np.maximum(mu2 - mu * mu, 1e-8)
        # Stima varianza rumore
        nv = auto_noise_variance(ch, ksize) if noise_var is None else float(noise_var)
        # Fattore di correzione
        g = np.maximum(var - nv, 0.0) / (var + 1e-8)
        out = mu + g * (ch - mu)
        res.append(out)
    out = cv2.merge(res)
    return to_uint8(out)



# Non-Local Means (NLM): riduzione rumore avanzata, confronta patch simili in tutta l'immagine.
#   Ogni pixel viene mediato con altri pixel simili anche lontani, preservando dettagli e texture.
# h: forza filtraggio, template_window/search_window: dimensioni finestre (forzate dispari).
def filtro_nlm(img: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    template_window = int(max(3, template_window) // 2 * 2 + 1)
    search_window = int(max(7, search_window) // 2 * 2 + 1)
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, h=float(h), templateWindowSize=template_window, searchWindowSize=search_window)
    else:
        return cv2.fastNlMeansDenoisingColored(img, h=float(h), hColor=float(h), templateWindowSize=template_window, searchWindowSize=search_window)



# Filtro Laplaciano: sharpening classico, evidenzia bordi tramite operatore Laplaciano.
#   Usa la seconda derivata (Laplaciano) per evidenziare le zone di rapido cambiamento d'intensità (bordi).
# alpha: intensità, ksize: dimensione kernel (forzata dispari).
def filtro_laplaciano(img: np.ndarray, alpha: float = 1.0, ksize: int = 3) -> np.ndarray:
    ksize = int(max(1, ksize) // 2 * 2 + 1)
    img_c = ensure_color(img)
    img_f = to_float(img_c)
    out_channels = []
    for ch in cv2.split(img_f):
        lap = cv2.Laplacian(ch, ddepth=cv2.CV_32F, ksize=ksize)
        out_ch = np.clip(ch - alpha * lap, 0.0, 1.0)
        out_channels.append(out_ch)
    out = cv2.merge(out_channels)
    return to_uint8(out)



# Unsharp Masking: sharpening tramite sottrazione di una versione sfocata (maschera di contrasto).
#   Aumenta la nitidezza sottraendo una versione sfocata dall'originale, enfatizzando i dettagli.
# radius: raggio blur, amount: intensità effetto.
def filtro_unsharp(img: np.ndarray, radius: int = 5, amount: float = 1.5) -> np.ndarray:
    radius = int(max(1, radius) // 2 * 2 + 1)
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)
    img_f = to_float(ensure_color(img))
    blurred_f = to_float(ensure_color(blurred))
    mask = img_f - blurred_f
    out = np.clip(img_f + amount * mask, 0.0, 1.0)
    return to_uint8(out)



# High-Boost Filtering: sharpening avanzato, amplifica dettagli mantenendo la base dell'immagine.
#   Simile all'unsharp, ma amplifica l'immagine originale prima di sottrarre la componente sfocata, esaltando micro-contrasto.
# k: fattore di amplificazione, radius: raggio blur per la base low-pass.
def filtro_high_boost(img: np.ndarray, radius: int = 5, k: float = 1.5) -> np.ndarray:
    radius = int(max(1, radius) // 2 * 2 + 1)
    low = cv2.GaussianBlur(img, (radius, radius), 0)
    img_f = to_float(ensure_color(img))
    low_f = to_float(ensure_color(low))
    out = np.clip(k * img_f - low_f, 0.0, 1.0)
    return to_uint8(out)



# Sobel/Prewitt: edge enhancement, evidenzia bordi tramite operatori di derivata.
#   Calcolano la derivata prima (gradiente) per evidenziare i bordi orizzontali e verticali.
# metodo: "sobel" o "prewitt", alpha: intensità effetto.
def filtro_sobel_prewitt(img: np.ndarray, metodo: str = "sobel", alpha: float = 1.0) -> np.ndarray:
    gray = ensure_gray(img)
    if metodo.lower() == "prewitt":
        # Kernel Prewitt
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        ky = kx.T
        gx = cv2.filter2D(gray, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    else:
        # Kernel Sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    # Edge enhancement: aggiunge i bordi all'immagine a colori
    color = ensure_color(img).astype(np.float32) / 255.0
    edge3 = cv2.merge([mag, mag, mag])
    out = np.clip(color + alpha * (edge3 - 0.5), 0.0, 1.0)
    return to_uint8(out)



# High-pass in frequenza: evidenzia dettagli tramite filtraggio nel dominio della frequenza (FFT).
#   Rimuove le componenti a bassa frequenza (aree uniformi), lasciando passare solo i dettagli e i bordi.
# cutoff: frequenza di taglio, alpha: intensità effetto.
def filtro_high_pass_freq(img: np.ndarray, cutoff: float = 30.0, alpha: float = 1.0) -> np.ndarray:
    color = ensure_color(img)
    hpf_channels = []
    for ch in cv2.split(color):
        # Trasformata di Fourier
        f = np.fft.fft2(ch.astype(np.float32))
        fshift = np.fft.fftshift(f)
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2
        # Costruzione filtro gaussiano high-pass
        y, x = np.ogrid[:rows, :cols]
        distance2 = (y - crow) ** 2 + (x - ccol) ** 2
        sigma2 = (cutoff ** 2)
        lpf = np.exp(-distance2 / (2.0 * sigma2 + 1e-8))
        hpf = 1.0 - lpf
        f_filtered = fshift * hpf
        ishift = np.fft.ifftshift(f_filtered)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)
        # Normalizza e aggiunge dettaglio all'originale
        img_hp_norm = cv2.normalize(img_back, None, 0.0, 1.0, cv2.NORM_MINMAX)
        base = ch.astype(np.float32) / 255.0
        ch_out = np.clip(base + alpha * (img_hp_norm - 0.5), 0.0, 1.0)
        hpf_channels.append((ch_out * 255.0).astype(np.uint8))
    return cv2.merge(hpf_channels)



# Deconvoluzione Richardson-Lucy: recupera nitidezza da blur di convoluzione.
#   Algoritmo iterativo di deconvoluzione che stima l'immagine originale conoscendo la funzione di sfocatura (PSF).
# iterations: step di iterazione, psf_size/psf_sigma definiscono PSF gaussiana approssimata.
def filtro_deconvoluzione_rl(
    img: np.ndarray,
    iterations: int = 15,
    psf_size: int = 7,
    psf_sigma: float = 1.6,
) -> np.ndarray:
    iterations = max(1, int(iterations))
    psf_size = int(max(3, psf_size) // 2 * 2 + 1)
    psf_sigma = float(max(0.1, psf_sigma))
    kernel_1d = cv2.getGaussianKernel(psf_size, psf_sigma)
    psf = kernel_1d @ kernel_1d.T
    psf = (psf / np.sum(psf)).astype(np.float32)
    psf_flip = psf[::-1, ::-1]

    color = ensure_color(img)
    restored = []
    for ch in cv2.split(to_float(color)):
        estimate = np.clip(ch.copy(), 1e-6, 1.0)
        for _ in range(iterations):
            conv_est = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
            conv_est = np.clip(conv_est, 1e-6, 1.0)
            relative_blur = ch / conv_est
            estimate *= cv2.filter2D(relative_blur, -1, psf_flip, borderType=cv2.BORDER_REFLECT)
            estimate = np.clip(estimate, 1e-6, 1.0)
        restored.append(estimate)
    out = cv2.merge(restored)
    return to_uint8(out)


# CLAHE su luminanza: aumenta contrasto locale agendo sulla componente L*a*b*.
#   Equalizzazione adattiva dell'istogramma che migliora il contrasto locale senza saturare il rumore.
def filtro_clahe_luminanza(img: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    clip_limit = float(max(1.0, clip_limit))
    tile = int(max(2, tile_grid))
    color = ensure_color(img)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# Retinex/Homomorphic: combina log-space e filtraggio gaussiano per equalizzare illuminazione.
#   Modella la riflettanza e l'illuminazione separatamente, correggendo ombre e non-uniformità di luce.
def filtro_retinex_homomorphic(
    img: np.ndarray,
    sigma: float = 40.0,
    gain: float = 2.0,
    offset: float = 0.0,
) -> np.ndarray:
    sigma = float(max(1.0, sigma))
    gain = float(max(0.1, gain))
    color = ensure_color(img)
    img_f = to_float(color)
    log_img = np.log1p(img_f)
    base = cv2.GaussianBlur(log_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    enhanced = gain * (log_img - base)
    enhanced = np.expm1(enhanced)
    enhanced = np.clip(enhanced + offset, 0.0, 1.0)
    return to_uint8(enhanced)


# Correzione colore + gamma: bilanciamento cromatico semplice e correzione gamma.
#   Bilancia i canali RGB per correggere dominanti e applica una correzione non lineare (gamma) per luminosità/contrasto.
def filtro_correzione_colore_gamma(
    img: np.ndarray,
    gamma: float = 1.2,
    balance_strength: float = 0.8,
) -> np.ndarray:
    gamma = float(max(0.1, gamma))
    balance_strength = float(max(0.0, min(1.0, balance_strength)))
    color = ensure_color(img)
    img_f = to_float(color)
    if balance_strength > 0.0:
        mean_rgb = np.mean(img_f.reshape(-1, 3), axis=0) + 1e-6
        gray_mean = float(np.mean(mean_rgb))
        gains = gray_mean / mean_rgb
        gains = 1.0 + balance_strength * (gains - 1.0)
        img_f = np.clip(img_f * gains, 0.0, 1.0)
    img_gamma = np.power(np.clip(img_f, 0.0, 1.0), 1.0 / gamma)
    return to_uint8(img_gamma)


# Morfologia luminanza: enfatizza dettagli tramite top-hat/black-hat sulla componente luminanza.
#   Operazioni morfologiche che evidenziano dettagli chiari (top-hat) o scuri (black-hat) rispetto al contesto locale.
def filtro_morfologia_luminanza(
    img: np.ndarray,
    operazione: str = "tophat",
    kernel_size: int = 15,
    strength: float = 1.0,
) -> np.ndarray:
    kernel_size = int(max(3, kernel_size) // 2 * 2 + 1)
    strength = float(max(0.0, strength))
    color = ensure_color(img)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    op = operazione.lower()
    morph_op = cv2.MORPH_BLACKHAT if op == "blackhat" else cv2.MORPH_TOPHAT
    morph = cv2.morphologyEx(l, morph_op, kernel)

    l_f = l.astype(np.float32) / 255.0
    morph_f = morph.astype(np.float32) / 255.0
    if op == "blackhat":
        l_adj = np.clip(l_f - strength * morph_f, 0.0, 1.0)
    else:
        l_adj = np.clip(l_f + strength * morph_f, 0.0, 1.0)
    l_out = (l_adj * 255.0 + 0.5).astype(np.uint8)
    lab_out = cv2.merge([l_out, a, b])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)



# Esportazione delle funzioni principali del modulo
__all__ = [
    "filtro_mediano",
    "filtro_gaussiano",
    "filtro_bilaterale",
    "filtro_wiener",
    "filtro_nlm",
    "filtro_laplaciano",
    "filtro_unsharp",
    "filtro_high_boost",
    "filtro_sobel_prewitt",
    "filtro_high_pass_freq",
    "filtro_deconvoluzione_rl",
    "filtro_clahe_luminanza",
    "filtro_retinex_homomorphic",
    "filtro_correzione_colore_gamma",
    "filtro_morfologia_luminanza",
]
