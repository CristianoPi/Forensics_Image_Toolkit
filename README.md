# Digital Forensics – CLI Forense per Immagini

Uno strumento da riga di comando per l’analisi e il miglioramento forense di immagini. 
Include filtri di riduzione rumore e sharpening, metriche quantitative sull’immagine e un sistema di suggerimenti (AI/euristico) che propone una catena di filtri con parametri adeguati. 

## Perché usarlo
- Migliorare la leggibilità di foto, frame di video o screenshot in contesti investigativi/forensi.
- Applicare filtri robusti preservando bordi e dettagli.
- Ottenere suggerimenti automatici in base a metriche oggettive dell’immagine.
- Avere un’architettura modulare e facilmente estendibile.

## Caratteristiche
- Caricamento immagine e validazione (grayscale, BGR, BGRA con gestione alfa).
- Riduzione rumore: mediano, gaussiano, bilaterale, Wiener (adattivo), Non‑Local Means.
- Sharpen mirati: Laplaciano, Unsharp Masking, High‑Boost.
- Filtri specializzati per recupero qualità: deconvoluzione Richardson‑Lucy, CLAHE su luminanza, Retinex/homomorphic, correzione colore & gamma, morfologia top/black-hat, oltre ai classici edge enhancement (Sobel/Prewitt, high-pass in frequenza).
- Metriche quantitative: nitidezza (varianza Laplaciano), stima rumore, luminosità, contrasto, range dinamico, densità bordi, colorfulness, rumore impulsivo.
- Suggerimenti automatici: chiama un modello OpenAI (opzionale) o usa euristiche locali; valida i parametri e applica i filtri in catena.
- Salvataggio output con timestamp in `outputs/` e suffissi descrittivi per ogni step.

## Architettura 
- Core:
  - `forensic_cli/core/image_utils.py` – conversioni, grayscale/color, stima rumore, timestamp.
  - `forensic_cli/core/filters.py` – implementazione dei filtri (tutti i trasformatori immagine).
  - `forensic_cli/core/metrics.py` – calcolo metriche oggettive.
- Services:
  - `forensic_cli/services/suggestions.py` – prompt + chiamata AI opzionale, fallback euristico, validazione parametri, applicazione singolo suggerimento.
  - `forensic_cli/services/io_service.py` – salvataggio output con naming consistente.
- Presentation (UI/CLI):
  - `forensic_cli/presentation/menu.py` – sistema menu e navigazione.
  - `forensic_cli/presentation/actions.py` – azioni utente che orchestrano filtri/servizi.
- Stato applicazione:
  - `forensic_cli/session.py` – path, immagine originale/corrente, directory output.
- Entrypoint:
  - `forensic_cli/cli.py` e `forensic_cli/__main__.py` – avvio CLI e costruzione menu.

Questa separazione permette di:
- Testare i filtri e le metriche senza dipendere dall’interfaccia CLI.
- Estendere i casi d’uso (batch, API, GUI) riusando il core.
- Mantenere bassa la complessità e l’accoppiamento tra livelli.

## Installazione
Requisiti: Python 3.9+ consigliato.

Pacchetti principali:
```
pip install numpy opencv-python python-dotenv
```
Opzionale (per suggerimenti AI):
```
pip install openai
```
Se usi un file `.env`, imposta la chiave:
```
OPENAI_API_KEY=sk-...
```

## Esecuzione
Entrypoint consigliato:
```
python -m forensic_cli
```

All’avvio puoi:
- Inserire il path dell’immagine oppure saltare e caricarla dal menu.
- Navigare tra i sottomenu: Riduzione del rumore, Sharpening, Altri filtri (deconvoluzione, CLAHE, Retinex, morfologia, edge enhancement).
- Vedere info, salvare corrente, ripristinare originale, chiedere suggerimenti AI/euristici.

Gli output vengono salvati in `outputs/` con pattern:
```
<basename>_<suffix>_<YYYYMMDD_HHMMSS>.<ext>
```
Se il file sorgente non ha estensione riconosciuta, l’output usa `.png`.

## Suggerimenti AI: come funziona
- Il servizio raccoglie metriche oggettive e costruisce un prompt con tutta la libreria di filtri e i range consentiti.
- Se la variabile `OPENAI_API_KEY` è presente, viene chiamato il modello (default: `gpt-4o-mini` per i costi) con una temperatura leggermente più alta per ottenere risposte meno ripetitive. Il modello deve rispondere con JSON valido.
- Se la chiamata fallisce o non è disponibile la rete/chiave, viene usato un fallback euristico basato sulle metriche (rumore, nitidezza, ecc.) limitato a filtri di riduzione rumore e sharpening.
- In entrambi i casi, i parametri vengono clippati ai range consentiti e gli step possono essere applicati in catena (senza limite rigido a 3 filtri), salvando un file per ogni step.

**Nota privacy**: l’immagine NON viene inviata al modello; si inviano solo metriche numeriche. Puoi verificare/estendere il comportamento in `forensic_cli/services/suggestions.py`.

## Metriche calcolate
- `var_laplacian`: nitidezza (varianza del Laplaciano).
- `noise_var`: stima del rumore (varianza locale media).
- `mean_brightness`, `std_contrast`: luminanza media e contrasto su scala di grigi.
- `dynamic_range`: (max-min)/255.
- `edges_ratio`: densità di bordi (Canny su soglie derivate dalla mediana).
- `colorfulness`: stima percepita della vividezza cromatica (solo immagini a 3 canali).
- `impulse_noise_ratio`: frazione di pixel quasi neri/quasi bianchi (sale/pepe).

## Filtri disponibili
- Rumore: `median`, `gaussian`, `bilateral`, `wiener`, `nlm`.
- Sharpen: `laplacian`, `unsharp`, `highboost`.
- Edge enhancement: `sobel`, `prewitt`, `highpass_freq`.
- Recupero qualità avanzato: `deconvolution_rl`, `clahe_luminance`, `retinex_homomorphic`, `color_gamma`, `morph_luminance`.

Le implementazioni si trovano in `forensic_cli/core/filters.py`.

## Estendere il progetto
- Aggiungere un filtro:
  1. Implementa la funzione in `forensic_cli/core/filters.py` (stateless, input/output `numpy.ndarray`).
  2. Se vuoi includerlo nei suggerimenti, aggiungi entry e range in `forensic_cli/services/suggestions.py` (`AVAILABLE_FILTERS`) e gestiscilo in `apply_single_suggestion`.
  3. Aggiungi un’azione CLI in `forensic_cli/presentation/actions.py` e un item nel menu in `forensic_cli/cli.py`.
- Modificare l’euristica: aggiorna `_heuristic_suggestions` e i relativi range.
- Cambiare naming output: modifica `forensic_cli/services/io_service.py`.

## Struttura del repository
```
forensic_cli/
  __init__.py
  __main__.py
  cli.py
  session.py
  core/
    __init__.py
    image_utils.py
    filters.py
    metrics.py
  services/
    __init__.py
    io_service.py
    suggestions.py
  presentation/
    __init__.py
    menu.py
    actions.py
outputs/               # output degli elaborati (generata a runtime)
```

## Note forensi e buone pratiche
- Conserva sempre il file originale. L’app non lo sovrascrive: lavora su copia in memoria e salva in una cartella separata.
- Documenta la catena di elaborazione: i suffissi degli output indicano i passaggi applicati e l’ordine.
- Preferisci parametri conservativi quando l’obiettivo è la leggibilità senza introdurre artefatti.
- Mantieni un log esterno delle operazioni se richiesto da policy/chain‑of‑custody.

## Problemi comuni
- ImportError OpenCV/Numpy: assicurati di avere `opencv-python` e `numpy` installati correttamente (eventuale ambiente virtuale).
- NLM lento su immagini grandi: riduci `search_window`/`template_window` o usa filtri più leggeri.
- Output vuoti: verifica i path e i permessi su `outputs/`.
- AI non disponibile: senza `OPENAI_API_KEY` o rete, il sistema usa automaticamente le euristiche locali.

## Sviluppi futuri
- Modalità non interattiva (CLI args) per pipeline batch riproducibili.
- Plugin di filtri e preset per domini specifici (documenti, targhe, volti sfocati).
- Report HTML con metriche pre/post e miniature.
- Supporto metadati (EXIF) e hash sui file in input/output.
- Test automatici per core/filters e core/metrics.

---
