# app/blueprints/main.py  (drop-in)
import os, io, csv, base64, json
from flask import Blueprint, render_template, request, current_app, flash
from werkzeug.utils import secure_filename
from joblib import load
from ..utils import extract_any, top_contributors

bp = Blueprint("main", __name__)

_model_cache = None
_meta_cache = None
_best_threshold = 0.5

def load_meta():
    global _meta_cache, _best_threshold
    # models/model_meta.json preferred; else metrics.json
    app_root = os.path.dirname(os.path.dirname(__file__))
    meta_path = os.path.join(app_root, "..", "models", "model_meta.json")
    metrics_path = os.path.join(app_root, "..", "models", "metrics.json")
    path = meta_path if os.path.exists(meta_path) else metrics_path
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _meta_cache = json.load(f)
            _best_threshold = float(_meta_cache.get("best_threshold", 0.5))

def get_model():
    global _model_cache
    if _model_cache is None:
        model_path = current_app.config["MODEL_PATH"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")
        _model_cache = load(model_path)
        load_meta()
    return _model_cache

@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@bp.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("pdfs")
    if not files or files[0].filename == "":
        flash("Please select at least one file (.pdf or .eml).")
        return render_template("index.html")

    model = get_model()
    results = []

    for f in files:
        filename = secure_filename(f.filename)
        try:
            text = extract_any(f)
            if not text.strip():
                results.append({"filename": filename, "error": "No extractable text found."})
                continue

            label = None
            proba = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba([text])[0]  # [ham, spam]
                proba = {"ham": float(p[0]), "spam": float(p[1])}
                label = "SPAM" if proba["spam"] >= _best_threshold else "Non-SPAM"
            else:
                pred = model.predict([text])[0]
                label = "SPAM" if int(pred) == 1 else "Non-SPAM"

            why = top_contributors(model, text, k=6)
            results.append({
                "filename": filename,
                "label": label,
                "proba": proba,
                "preview": text[:800],
                "why": why
            })
        except Exception as e:
            results.append({"filename": filename, "error": str(e)})

    # Batch summary + CSV export
    total = len(results)
    spam = sum(1 for r in results if r.get("label") == "SPAM")
    ham = sum(1 for r in results if r.get("label") == "Non-SPAM")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["filename","label","spam_prob"])
    w.writeheader()
    for r in results:
        w.writerow({
            "filename": r.get("filename"),
            "label": r.get("label"),
            "spam_prob": (r.get("proba") or {}).get("spam")
        })
    dl_csv = base64.b64encode(buf.getvalue().encode()).decode()

    return render_template("results.html",
                           results=results, total=total, spam=spam, ham=ham, dl_csv=dl_csv,
                           threshold=_best_threshold, meta=_meta_cache)
