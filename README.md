# SpamGuard Pro — PDF Email Spam Classifier

A professional, single‑page Flask app that classifies **PDF emails** as **SPAM** or **Non‑SPAM** using a classic ML pipeline (TF‑IDF + Linear SVM / Logistic Regression). Trains either from the **Kaggle preprocessed TREC07p CSV** or the **raw TREC07p** index.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # optional
```

### Train (Fast path — Kaggle CSV)
```bash
python train_csv.py --csv /path/to/trec07p_preprocessed.csv --model linear_svc
# or with probabilities in UI:
python train_csv.py --csv /path/to/trec07p_preprocessed.csv --model logreg
```
Artifacts written to `models/`:
- `spam_pipeline.joblib`
- `metrics.json`

### Train (Raw TREC07p)
```bash
python train_trec07p.py --data_dir /path/to/trec07p --subset 15000 --model linear_svc
```

### Run the app
```bash
python run.py
# open http://localhost:5000
```

## Features
- Modern, accessible UI with drag‑and‑drop.
- **Batch PDF** upload; per‑file label + (optional) confidence.
- Robust PDF text extraction via PyPDF2 (no OCR).
- Clean separation: app factory, blueprint, utils, training scripts.

## Notes & Limitations
- Scanned PDFs (images) require OCR (e.g., pytesseract) — not included to keep dependencies lean.
- Dataset is 2007-era; report discusses domain shift / future improvements.

## License
For academic coursework use.
