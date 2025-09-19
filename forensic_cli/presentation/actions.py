#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

from typing import Optional

import numpy as np

from forensic_cli.session import Session
from forensic_cli.core.filters import (
    filtro_mediano,
    filtro_gaussiano,
    filtro_bilaterale,
    filtro_wiener,
    filtro_nlm,
    filtro_laplaciano,
    filtro_unsharp,
    filtro_high_boost,
    filtro_sobel_prewitt,
    filtro_high_pass_freq,
    filtro_deconvoluzione_rl,
    filtro_clahe_luminanza,
    filtro_retinex_homomorphic,
    filtro_correzione_colore_gamma,
    filtro_morfologia_luminanza,
)
from forensic_cli.core.metrics import compute_image_metrics
from forensic_cli.services.io_service import salva_output
from forensic_cli.services.suggestions import suggest_filters, apply_single_suggestion


def info_img(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    return f"{w}x{h} px, {ch} canali, dtype={img.dtype}"


def ask_int(prompt: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            val = int(s)
            if min_val is not None and val < min_val:
                print(f"Valore minimo {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Valore massimo {max_val}.")
                continue
            return val
        except ValueError:
            print("Inserire un intero valido.")


def ask_float(prompt: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            val = float(s)
            if min_val is not None and val < min_val:
                print(f"Valore minimo {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Valore massimo {max_val}.")
                continue
            return val
        except ValueError:
            print("Inserire un numero valido.")


def azione_carica_immagine(session: Session):
    path = input("Inserisci il path dell'immagine: ").strip().strip('"').strip("'")
    if session.load_image(path):
        print("Verifica apertura: OK")


def azione_info_immagine(session: Session):
    if not session.require_image():
        return
    assert session.current is not None
    print(f"Immagine corrente: {info_img(session.current)}")


def azione_salva_corrente(session: Session, suffix: str = "manuale"):
    if not session.require_image():
        return
    out = salva_output(session.current, session.img_path or "immagine.png", suffix, session.out_dir)
    print(f"Salvato: {out}")


def azione_ripristina_originale(session: Session):
    if session.original is None:
        print("Nessuna immagine da ripristinare.")
        return
    session.current = session.original.copy()
    print("Immagine ripristinata all'originale.")


# --- Azioni filtri riduzione rumore --- #


def azione_filtro_mediano(session: Session):
    if not session.require_image():
        return
    k = ask_int("Dimensione kernel (dispari)", 3, 3)
    session.current = filtro_mediano(session.current, k)
    out = salva_output(session.current, session.img_path or "img.png", f"median_k{k}", session.out_dir)
    print(f"Applicato Filtro Mediano. Output: {out}")


def azione_filtro_gaussiano(session: Session):
    if not session.require_image():
        return
    k = ask_int("Dimensione kernel (dispari)", 5, 3)
    s = ask_float("Sigma (0=auto)", 0.0, 0.0)
    session.current = filtro_gaussiano(session.current, k, s)
    out = salva_output(session.current, session.img_path or "img.png", f"gauss_k{k}_s{s}", session.out_dir)
    print(f"Applicato Filtro Gaussiano. Output: {out}")


def azione_filtro_bilaterale(session: Session):
    if not session.require_image():
        return
    d = ask_int("Diametro pixel", 9, 1)
    sc = ask_float("Sigma Color", 75.0, 0.1)
    ss = ask_float("Sigma Space", 75.0, 0.1)
    session.current = filtro_bilaterale(session.current, d, sc, ss)
    out = salva_output(session.current, session.img_path or "img.png", f"bilateral_d{d}_sc{sc}_ss{ss}", session.out_dir)
    print(f"Applicato Filtro Bilaterale. Output: {out}")


def azione_filtro_wiener(session: Session):
    if not session.require_image():
        return
    k = ask_int("Dimensione finestra", 5, 3)
    nv = input("Varianza rumore (vuoto = auto): ").strip()
    noise_var = float(nv) if nv != "" else None
    session.current = filtro_wiener(session.current, k, noise_var)
    suffix = f"wiener_k{k}_nv{noise_var if noise_var is not None else 'auto'}"
    out = salva_output(session.current, session.img_path or "img.png", suffix, session.out_dir)
    print(f"Applicato Filtro Wiener. Output: {out}")


def azione_filtro_nlm(session: Session):
    if not session.require_image():
        return
    h = ask_float("Forza filtraggio h", 10.0, 0.0)
    tw = ask_int("Finestra template (dispari)", 7, 3)
    sw = ask_int("Finestra ricerca (dispari)", 21, 7)
    session.current = filtro_nlm(session.current, h, tw, sw)
    out = salva_output(session.current, session.img_path or "img.png", f"nlm_h{h}_tw{tw}_sw{sw}", session.out_dir)
    print(f"Applicato Non-Local Means. Output: {out}")


# --- Azioni filtri sharpening --- #


def azione_filtro_laplaciano(session: Session):
    if not session.require_image():
        return
    a = ask_float("Intensità (alpha)", 1.0)
    k = ask_int("Kernel Laplaciano (dispari)", 3, 1)
    session.current = filtro_laplaciano(session.current, a, k)
    out = salva_output(session.current, session.img_path or "img.png", f"laplacian_a{a}_k{k}", session.out_dir)
    print(f"Applicato Laplaciano. Output: {out}")


def azione_unsharp(session: Session):
    if not session.require_image():
        return
    r = ask_int("Raggio blur (dispari)", 5, 1)
    a = ask_float("Amount", 1.5, 0.0)
    session.current = filtro_unsharp(session.current, r, a)
    out = salva_output(session.current, session.img_path or "img.png", f"unsharp_r{r}_a{a}", session.out_dir)
    print(f"Applicata Unsharp Mask. Output: {out}")


def azione_high_boost(session: Session):
    if not session.require_image():
        return
    r = ask_int("Raggio low-pass (dispari)", 5, 1)
    k = ask_float("Fattore k", 1.5, 0.0)
    session.current = filtro_high_boost(session.current, r, k)
    out = salva_output(session.current, session.img_path or "img.png", f"highboost_r{r}_k{k}", session.out_dir)
    print(f"Applicato High-Boost. Output: {out}")


def azione_sobel_prewitt(session: Session):
    if not session.require_image():
        return
    metodo = input("Metodo (sobel/prewitt) [sobel]: ").strip().lower() or "sobel"
    a = ask_float("Intensità enhancement (alpha)", 1.0)
    session.current = filtro_sobel_prewitt(session.current, metodo, a)
    out = salva_output(session.current, session.img_path or "img.png", f"{metodo}_a{a}", session.out_dir)
    print(f"Applicato {metodo.title()} edge enhancement. Output: {out}")


def azione_high_pass_freq(session: Session):
    if not session.require_image():
        return
    c = ask_float("Cutoff (sigma in frequenza)", 30.0, 1.0)
    a = ask_float("Intensità enhancement (alpha)", 1.0)
    session.current = filtro_high_pass_freq(session.current, c, a)
    out = salva_output(session.current, session.img_path or "img.png", f"hpfreq_c{c}_a{a}", session.out_dir)
    print(f"Applicato High-Pass in frequenza. Output: {out}")


# --- Azioni altri filtri specializzati --- #


def azione_deconvoluzione_rl(session: Session):
    if not session.require_image():
        return
    iters = ask_int("Iterazioni Richardson-Lucy", 20, 1, 100)
    psf = ask_int("Dimensione PSF gaussiana (dispari)", 7, 3)
    sigma = ask_float("Sigma PSF", 1.8, 0.1)
    session.current = filtro_deconvoluzione_rl(session.current, iters, psf, sigma)
    suffix = f"rl_it{iters}_psf{psf}_sg{sigma}"
    out = salva_output(session.current, session.img_path or "img.png", suffix, session.out_dir)
    print(f"Applicata deconvoluzione Richardson-Lucy. Output: {out}")


def azione_clahe_luminanza(session: Session):
    if not session.require_image():
        return
    clip = ask_float("Clip limit CLAHE", 3.0, 1.0)
    tile = ask_int("Dimensione tile", 8, 2)
    session.current = filtro_clahe_luminanza(session.current, clip, tile)
    out = salva_output(session.current, session.img_path or "img.png", f"clahe_clip{clip}_tile{tile}", session.out_dir)
    print(f"Applicato CLAHE su luminanza. Output: {out}")


def azione_retinex_homomorphic(session: Session):
    if not session.require_image():
        return
    sigma = ask_float("Sigma gaussiana (illuminazione)", 35.0, 1.0)
    gain = ask_float("Gain Retinex", 2.2, 0.1)
    offset = ask_float("Offset finale", 0.0, -0.5, 0.5)
    session.current = filtro_retinex_homomorphic(session.current, sigma, gain, offset)
    suffix = f"retinex_s{sigma}_g{gain}_o{offset}"
    out = salva_output(session.current, session.img_path or "img.png", suffix, session.out_dir)
    print(f"Applicato filtro Retinex/Homomorphic. Output: {out}")


def azione_correzione_colore_gamma(session: Session):
    if not session.require_image():
        return
    gamma = ask_float("Gamma", 1.2, 0.1)
    balance = ask_float("Bilanciamento cromatico (0-1)", 0.8, 0.0, 1.0)
    session.current = filtro_correzione_colore_gamma(session.current, gamma, balance)
    suffix = f"colorcorr_g{gamma}_b{balance}"
    out = salva_output(session.current, session.img_path or "img.png", suffix, session.out_dir)
    print(f"Applicata correzione colore & gamma. Output: {out}")


def azione_morfologia_luminanza(session: Session):
    if not session.require_image():
        return
    op = input("Operazione (tophat/blackhat) [tophat]: ").strip().lower() or "tophat"
    kernel = ask_int("Kernel morfologico (dispari)", 15, 3)
    strength = ask_float("Intensità contributo", 1.0, 0.0, 3.0)
    session.current = filtro_morfologia_luminanza(session.current, op, kernel, strength)
    suffix = f"morph_{op}_k{kernel}_s{strength}"
    out = salva_output(session.current, session.img_path or "img.png", suffix, session.out_dir)
    print(f"Applicata morfologia luminanza ({op}). Output: {out}")


def _esegui_ai_suggerimenti(session: Session, user_hint: Optional[str], heading: str):
    if session.current is None:
        print("[INFO] Nessuna immagine caricata.")
        return

    assert session.current is not None
    clean_hint = user_hint.strip() if user_hint else ""
    guided = bool(clean_hint)
    status = "guidata" if guided else "in corso"
    print(f"\n[AI] Analisi {status} (metriche + eventuale chiamata al modello)...")
    if guided:
        print(f"Nota analista inoltrata: {clean_hint}")

    suggestions, metrics, source = suggest_filters(session.current, user_hint=clean_hint if guided else None)

    print("\nMetriche immagine (per AI):")
    import json as _json
    print(_json.dumps(metrics, indent=2))

    if not suggestions:
        print("Nessun suggerimento disponibile.")
        return

    origin = "[da AI]" if source == "ai" else "[euristiche]"
    print(f"\n{heading} {origin}:")
    for i, s in enumerate(suggestions, 1):
        print(f" {i}) {s['filter']} params={s['params']} conf={s.get('confidence', 0.0):.2f}")
        if s.get("rationale"):
            print(f"    → {s['rationale']}")

    choice = input("\nApplico automaticamente i suggerimenti? (s/N): ").strip().lower()
    if choice != 's':
        return

    current = session.current
    for i, s in enumerate(suggestions, 1):
        current = apply_single_suggestion(current, s)
        suffix = f"ai_step{i}_{s['filter']}"
        out = salva_output(current, session.img_path or "img.png", suffix, session.out_dir)
        print(f"Salvato step {i}: {out}")
    session.current = current
    print("Applicazione suggerimenti completata.")


def azione_ai_suggerimenti(session: Session):
    _esegui_ai_suggerimenti(session, user_hint=None, heading="Suggerimenti (ordine proposto)")


def azione_ai_guidata(session: Session):
    if session.current is None:
        print("[INFO] Nessuna immagine caricata.")
        return

    prompt = input(
        "Inserisci una nota per l'AI (descrivi contesto e obiettivo, vuoto per annullare): "
    ).strip()
    if prompt == "":
        print("Operazione annullata: nessuna nota fornita.")
        return

    _esegui_ai_suggerimenti(
        session,
        user_hint=prompt,
        heading="Suggerimenti guidati (ordine proposto)",
    )


__all__ = [
    "azione_carica_immagine",
    "azione_info_immagine",
    "azione_salva_corrente",
    "azione_ripristina_originale",
    "azione_filtro_mediano",
    "azione_filtro_gaussiano",
    "azione_filtro_bilaterale",
    "azione_filtro_wiener",
    "azione_filtro_nlm",
    "azione_filtro_laplaciano",
    "azione_unsharp",
    "azione_high_boost",
    "azione_sobel_prewitt",
    "azione_high_pass_freq",
    "azione_deconvoluzione_rl",
    "azione_clahe_luminanza",
    "azione_retinex_homomorphic",
    "azione_correzione_colore_gamma",
    "azione_morfologia_luminanza",
    "azione_ai_suggerimenti",
    "azione_ai_guidata",
]
