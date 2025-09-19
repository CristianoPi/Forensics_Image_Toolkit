#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from forensic_cli.core.metrics import compute_image_metrics
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


AVAILABLE_FILTERS = {
    "median": {
        "label": "Filtro Mediano",
        "params": {"ksize": {"type": "int", "min": 3, "max": 15, "odd": True}},
    },
    "gaussian": {
        "label": "Filtro Gaussiano",
        "params": {
            "ksize": {"type": "int", "min": 3, "max": 31, "odd": True},
            "sigma": {"type": "float", "min": 0.0, "max": 5.0},
        },
    },
    "bilateral": {
        "label": "Filtro Bilaterale",
        "params": {
            "d": {"type": "int", "min": 1, "max": 25},
            "sigma_color": {"type": "float", "min": 5.0, "max": 200.0},
            "sigma_space": {"type": "float", "min": 5.0, "max": 200.0},
        },
    },
    "wiener": {
        "label": "Filtro Wiener (adattivo)",
        "params": {"ksize": {"type": "int", "min": 3, "max": 15, "odd": True}, "noise_var": {"type": "float", "min": 0.0, "max": 0.01}},
    },
    "nlm": {
        "label": "Non-Local Means",
        "params": {
            "h": {"type": "float", "min": 1.0, "max": 30.0},
            "template_window": {"type": "int", "min": 3, "max": 21, "odd": True},
            "search_window": {"type": "int", "min": 7, "max": 31, "odd": True},
        },
    },
    "laplacian": {
        "label": "Laplaciano",
        "params": {"alpha": {"type": "float", "min": 0.1, "max": 3.0}, "ksize": {"type": "int", "min": 1, "max": 7, "odd": True}},
    },
    "unsharp": {
        "label": "Unsharp Masking",
        "params": {"radius": {"type": "int", "min": 1, "max": 15, "odd": True}, "amount": {"type": "float", "min": 0.2, "max": 3.0}},
    },
    "highboost": {
        "label": "High-Boost",
        "params": {"radius": {"type": "int", "min": 1, "max": 15, "odd": True}, "k": {"type": "float", "min": 1.0, "max": 3.0}},
    },
    "sobel": {
        "label": "Sobel (edge enhance)",
        "params": {"alpha": {"type": "float", "min": 0.2, "max": 3.0}},
    },
    "prewitt": {
        "label": "Prewitt (edge enhance)",
        "params": {"alpha": {"type": "float", "min": 0.2, "max": 3.0}},
    },
    "highpass_freq": {
        "label": "High-Pass (frequenza)",
        "params": {"cutoff": {"type": "float", "min": 1.0, "max": 200.0}, "alpha": {"type": "float", "min": 0.2, "max": 3.0}},
    },
    "deconvolution_rl": {
        "label": "Deconvoluzione Richardson-Lucy",
        "params": {
            "iterations": {"type": "int", "min": 5, "max": 60},
            "psf_size": {"type": "int", "min": 3, "max": 21, "odd": True},
            "psf_sigma": {"type": "float", "min": 0.5, "max": 5.0},
        },
    },
    "clahe_luminance": {
        "label": "CLAHE su luminanza",
        "params": {
            "clip_limit": {"type": "float", "min": 1.0, "max": 6.0},
            "tile_grid": {"type": "int", "min": 2, "max": 16},
        },
    },
    "retinex_homomorphic": {
        "label": "Retinex / Homomorphic",
        "params": {
            "sigma": {"type": "float", "min": 5.0, "max": 80.0},
            "gain": {"type": "float", "min": 0.5, "max": 4.0},
            "offset": {"type": "float", "min": -0.3, "max": 0.3},
        },
    },
    "color_gamma": {
        "label": "Correzione colore & Gamma",
        "params": {
            "gamma": {"type": "float", "min": 0.5, "max": 2.5},
            "balance_strength": {"type": "float", "min": 0.0, "max": 1.0},
        },
    },
    "morph_luminance": {
        "label": "Morfologia luminanza",
        "params": {
            "operation": {"type": "str", "choices": ["tophat", "blackhat"]},
            "kernel_size": {"type": "int", "min": 5, "max": 41, "odd": True},
            "strength": {"type": "float", "min": 0.1, "max": 3.0},
        },
    },
}


def _build_ai_prompt(metrics: dict, user_hint: Optional[str] = None) -> str:
    filt_desc = []
    for key, meta in AVAILABLE_FILTERS.items():
        params_desc = []
        for p, spec in meta["params"].items():
            p_type = spec.get("type")
            if p_type in {"int", "float"}:
                min_val = spec.get("min")
                max_val = spec.get("max")
                odd = " (dispari)" if spec.get("odd") else ""
                params_desc.append(f"{p}: {p_type} [{min_val}-{max_val}{odd}]")
            elif p_type == "str":
                choices = "/".join(spec.get("choices", []))
                params_desc.append(f"{p}: {p_type} {{{choices}}}")
            else:
                params_desc.append(f"{p}: {p_type}")
        filt_desc.append(f"- {key}: {', '.join(params_desc)}")
    filters_block = "\n".join(filt_desc)
    analyst_block = ""
    analyst_constraint = ""
    if user_hint:
        cleaned = user_hint.strip()
        if cleaned:
            analyst_block = f"\nNota dell'analista:\n{cleaned}\n"
            analyst_constraint = "- Considera attentamente la nota dell'analista per guidare le scelte.\n"
    prompt = f"""
Sei un assistente per miglioramento immagini forense. Ti fornisco metriche quantitative dell'immagine.
In base ad esse, suggerisci una catena ordinata di filtri (usa tutti quelli necessari) scegliendo tra l'intera libreria disponibile, con parametri pertinenti.
Rispondi SOLO con JSON valido con chiave `suggestions` (lista), ogni elemento con chiavi:
- filter: chiave tra {list(AVAILABLE_FILTERS.keys())}
- params: dizionario dei parametri
- rationale: breve spiegazione italiana
- confidence: numero 0..1

Metriche:
{json.dumps(metrics, indent=2)}

{analyst_block}
Filtri disponibili e parametri:
{filters_block}

Vincoli:
- Rispondi solo JSON, nessun testo extra.
- Usa parametri nei range, rispetta `odd` dove indicato.
- Considera prima il rumore e il recupero di nitidezza, poi valuta filtri avanzati (deconvoluzione, CLAHE, Retinex, morfologia, colore).
- Sii incisivo: opta per combinazioni credibili per immagini forensi difficili (sfocate, buie, compressioni forti).
- Se necessario, usa più di tre filtri e varia i parametri per evitare risultati ripetitivi.
{analyst_constraint}
""".strip()
    return prompt


def _try_openai_chat(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Rispondi sempre e solo con JSON valido."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
            )
            return resp.choices[0].message.content
        except Exception:
            import openai  # type: ignore
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Rispondi sempre e solo con JSON valido."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.35,
            )
            return resp.choices[0].message["content"]
    except Exception:
        return None


def _parse_suggestions_json(text: str) -> Optional[List[dict]]:
    if not text:
        return None
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict) and isinstance(obj.get("suggestions"), list):
            return obj["suggestions"]
        return None
    except Exception:
        return None


def _oddify(x: int) -> int:
    x = int(x)
    return max(1, x // 2 * 2 + 1)


def _clip_param(val, spec):
    t = spec.get("type")
    if t == "str":
        choices = [c.lower() for c in spec.get("choices", [])]
        if not choices:
            return str(val)
        candidate = str(val).lower()
        return choices[0] if candidate not in choices else candidate
    v = float(val)
    v = max(float(spec.get("min", v)), min(float(spec.get("max", v)), v))
    if t == "int":
        v = int(round(v))
        if spec.get("odd"):
            v = _oddify(v)
    return v


def _heuristic_suggestions(metrics: dict, is_color: bool) -> List[dict]:
    suggestions: List[dict] = []
    noise = metrics.get("noise_var", 0.0)
    blur = metrics.get("var_laplacian", 0.0)
    impulse = metrics.get("impulse_noise_ratio", 0.0)
    dyn = metrics.get("dynamic_range", 1.0)

    if impulse > 0.01:
        k = 5 if impulse < 0.03 else 7
        suggestions.append({
            "filter": "median",
            "params": {"ksize": k},
            "rationale": f"Presenza di rumore impulsivo ({impulse:.2%}). Mediano rimuove sale/pepe mantenendo bordi.",
            "confidence": 0.8,
        })
    elif noise > 0.0015:
        if is_color:
            suggestions.append({
                "filter": "nlm",
                "params": {"h": 15.0, "template_window": 9, "search_window": 27},
                "rationale": f"Rumore elevato (var={noise:.4f}). NLM deciso preservando dettagli a colori.",
                "confidence": 0.75,
            })
        else:
            suggestions.append({
                "filter": "wiener",
                "params": {"ksize": 7, "noise_var": round(noise, 5)},
                "rationale": f"Rumore simile a gaussiano (var={noise:.4f}). Wiener adattivo e più deciso.",
                "confidence": 0.7,
            })
    elif noise > 0.0006:
        suggestions.append({
            "filter": "bilateral",
            "params": {"d": 9, "sigma_color": 65.0, "sigma_space": 65.0},
            "rationale": f"Rumore moderato (var={noise:.4f}). Bilaterale deciso preserva i bordi.",
            "confidence": 0.65,
        })

    if blur < 20:
        suggestions.append({
            "filter": "unsharp",
            "params": {"radius": 7, "amount": 2.3},
            "rationale": f"Immagine molto morbida (varLap={blur:.1f}). Unsharp molto incisivo.",
            "confidence": 0.75,
        })
    elif blur < 60:
        suggestions.append({
            "filter": "unsharp",
            "params": {"radius": 5, "amount": 1.8},
            "rationale": f"Nitidezza bassa (varLap={blur:.1f}). Unsharp più decisa.",
            "confidence": 0.7,
        })
    elif dyn < 0.35:
        suggestions.append({
            "filter": "highboost",
            "params": {"radius": 5, "k": 1.6},
            "rationale": f"Basso range dinamico (DR={dyn:.2f}). High-boost più incisivo per micro-contrasto.",
            "confidence": 0.6,
        })

    return suggestions


def _validate_params(suggestions: List[dict]) -> List[dict]:
    validated = []
    for s in suggestions:
        fkey = s.get("filter")
        params = dict(s.get("params", {}))
        meta = AVAILABLE_FILTERS.get(fkey)
        if not meta:
            continue
        clean = {}
        for p, spec in meta["params"].items():
            if p in params:
                clean[p] = _clip_param(params[p], spec)
        validated.append({
            "filter": fkey,
            "params": clean,
            "rationale": s.get("rationale", ""),
            "confidence": float(s.get("confidence", 0.5)),
        })
    return validated


def suggest_filters(img: np.ndarray, user_hint: Optional[str] = None) -> Tuple[List[dict], dict, str]:
    """Return (suggestions, metrics, source)."""
    metrics = compute_image_metrics(img)

    prompt = _build_ai_prompt(metrics, user_hint=user_hint)
    source = "heuristic"

    text = _try_openai_chat(prompt)
    suggestions: Optional[List[dict]] = None
    if text:
        parsed = _parse_suggestions_json(text)
        if parsed:
            suggestions = parsed
            source = "ai"

    if suggestions is None:
        suggestions = _heuristic_suggestions(metrics, is_color=(img.ndim == 3))
        source = "heuristic"

    return _validate_params(suggestions), metrics, source


def apply_single_suggestion(img: np.ndarray, s: dict) -> np.ndarray:
    f = s.get("filter")
    p = s.get("params", {})
    if f == "median":
        return filtro_mediano(img, int(p.get("ksize", 3)))
    if f == "gaussian":
        return filtro_gaussiano(img, int(p.get("ksize", 5)), float(p.get("sigma", 0.0)))
    if f == "bilateral":
        return filtro_bilaterale(img, int(p.get("d", 9)), float(p.get("sigma_color", 75.0)), float(p.get("sigma_space", 75.0)))
    if f == "wiener":
        nv = p.get("noise_var", None)
        nv = float(nv) if nv is not None else None
        return filtro_wiener(img, int(p.get("ksize", 5)), nv)
    if f == "nlm":
        return filtro_nlm(img, float(p.get("h", 10.0)), int(p.get("template_window", 7)), int(p.get("search_window", 21)))
    if f == "laplacian":
        return filtro_laplaciano(img, float(p.get("alpha", 1.0)), int(p.get("ksize", 3)))
    if f == "unsharp":
        return filtro_unsharp(img, int(p.get("radius", 5)), float(p.get("amount", 1.5)))
    if f == "highboost":
        return filtro_high_boost(img, int(p.get("radius", 5)), float(p.get("k", 1.5)))
    if f == "sobel":
        return filtro_sobel_prewitt(img, "sobel", float(p.get("alpha", 1.0)))
    if f == "prewitt":
        return filtro_sobel_prewitt(img, "prewitt", float(p.get("alpha", 1.0)))
    if f == "highpass_freq":
        return filtro_high_pass_freq(img, float(p.get("cutoff", 30.0)), float(p.get("alpha", 1.0)))
    if f == "deconvolution_rl":
        return filtro_deconvoluzione_rl(
            img,
            int(p.get("iterations", 15)),
            int(p.get("psf_size", 7)),
            float(p.get("psf_sigma", 1.6)),
        )
    if f == "clahe_luminance":
        return filtro_clahe_luminanza(
            img,
            float(p.get("clip_limit", 3.0)),
            int(p.get("tile_grid", 8)),
        )
    if f == "retinex_homomorphic":
        return filtro_retinex_homomorphic(
            img,
            float(p.get("sigma", 40.0)),
            float(p.get("gain", 2.0)),
            float(p.get("offset", 0.0)),
        )
    if f == "color_gamma":
        return filtro_correzione_colore_gamma(
            img,
            float(p.get("gamma", 1.2)),
            float(p.get("balance_strength", 0.8)),
        )
    if f == "morph_luminance":
        op = str(p.get("operation", "tophat")).lower()
        return filtro_morfologia_luminanza(
            img,
            op,
            int(p.get("kernel_size", 15)),
            float(p.get("strength", 1.0)),
        )
    return img


__all__ = [
    "AVAILABLE_FILTERS",
    "suggest_filters",
    "apply_single_suggestion",
]
