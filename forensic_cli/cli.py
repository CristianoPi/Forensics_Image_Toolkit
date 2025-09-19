#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

from forensic_cli.presentation.menu import Menu
from forensic_cli.presentation.actions import (
    azione_carica_immagine,
    azione_info_immagine,
    azione_salva_corrente,
    azione_ripristina_originale,
    azione_filtro_mediano,
    azione_filtro_gaussiano,
    azione_filtro_bilaterale,
    azione_filtro_wiener,
    azione_filtro_nlm,
    azione_filtro_laplaciano,
    azione_unsharp,
    azione_high_boost,
    azione_sobel_prewitt,
    azione_high_pass_freq,
    azione_deconvoluzione_rl,
    azione_clahe_luminanza,
    azione_retinex_homomorphic,
    azione_correzione_colore_gamma,
    azione_morfologia_luminanza,
    azione_ai_suggerimenti,
    azione_ai_guidata,
)
from forensic_cli.session import Session


def build_menus() -> Menu:
    main = Menu("Menu Principale - Strumenti Forensi Immagini")

    sub_noise = Menu("Riduzione del rumore")
    sub_noise.add("1", "Filtro Mediano", action=azione_filtro_mediano)
    sub_noise.add("2", "Filtro Gaussiano", action=azione_filtro_gaussiano)
    sub_noise.add("3", "Filtro Bilaterale", action=azione_filtro_bilaterale)
    sub_noise.add("4", "Filtro Wiener (adattivo)", action=azione_filtro_wiener)
    sub_noise.add("5", "Non-Local Means (NLM)", action=azione_filtro_nlm)
    sub_noise.add("b", "Torna indietro")

    sub_sharp = Menu("Sharpening")
    sub_sharp.add("1", "Laplaciano", action=azione_filtro_laplaciano)
    sub_sharp.add("2", "Unsharp Masking", action=azione_unsharp)
    sub_sharp.add("3", "High-Boost", action=azione_high_boost)
    sub_sharp.add("b", "Torna indietro")

    sub_other = Menu("Altri filtri specializzati")
    sub_other.add("1", "Deconvoluzione Richardson-Lucy", action=azione_deconvoluzione_rl)
    sub_other.add("2", "CLAHE su luminanza", action=azione_clahe_luminanza)
    sub_other.add("3", "Retinex / Homomorphic", action=azione_retinex_homomorphic)
    sub_other.add("4", "Correzione colore & Gamma", action=azione_correzione_colore_gamma)
    sub_other.add("5", "Morfologia luminanza (Top/Black-hat)", action=azione_morfologia_luminanza)
    sub_other.add("6", "Sobel/Prewitt (edge enhancement)", action=azione_sobel_prewitt)
    sub_other.add("7", "High-Pass in frequenza", action=azione_high_pass_freq)
    sub_other.add("b", "Torna indietro")

    main.add("1", "Carica immagine", action=azione_carica_immagine)
    main.add("2", "Riduzione del rumore", submenu=sub_noise)
    main.add("3", "Sharpening", submenu=sub_sharp)
    main.add("4", "Altri filtri", submenu=sub_other)
    main.add("5", "Info immagine", action=azione_info_immagine)
    main.add("6", "Salva immagine corrente", action=lambda s: azione_salva_corrente(s, "save"))
    main.add("7", "Ripristina immagine originale", action=azione_ripristina_originale)
    main.add("8", "Suggerimenti AI (filtri e parametri)", action=azione_ai_suggerimenti)
    main.add("9", "Suggerimenti AI guidati (con prompt analista)", action=azione_ai_guidata)
    main.add("q", "Esci")
    return main


def banner():
    print("=" * 70)
    print("CLI Forense - Image Toolkit")
    print("Autori: Cristiano Pistorio, Sofia Manno - 2025")
    print("Dipendenze: numpy, opencv-python | Opzionale: OPENAI_API_KEY per AI")
    print("=" * 70)


def main():
    banner()
    session = Session()

    while True:
        path = input("Inserisci il path dell'immagine (o lascia vuoto per saltare): ").strip().strip('"').strip("'")
        if path == "":
            print("Nessuna immagine caricata all'avvio. Puoi caricarla dal menu.")
            break
        if session.load_image(path):
            break
        else:
            print("Impossibile caricare. Riprova.")

    menu = build_menus()
    menu.run(session)
    print("Uscita. Grazie per l'utilizzo.")


if __name__ == "__main__":
    main()
