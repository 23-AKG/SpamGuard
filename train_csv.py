# train_csv.py  (drop-in)
import argparse, json, time, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def build_pipeline(model: str = "linear_svc") -> Pipeline:
    tfidf = TfidfVectorizer(
        stop_words="english", lowercase=True, strip_accents="unicode",
        max_df=0.98, min_df=2, ngram_range=(1, 2)
    )
    if model == "linear_svc":
        # Calibrate to get predict_proba
        base = LinearSVC()
        clf = CalibratedClassifierCV(base, cv=3)
    elif model == "logreg":
        clf = LogisticRegression(max_iter=300)
    elif model == "nb":
        clf = MultinomialNB()
    else:
        raise ValueError("model must be one of: linear_svc, logreg, nb")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Kaggle preprocessed CSV")
    ap.add_argument("--text_cols", default="subject,message", help="Comma-separated text columns to concatenate")
    ap.add_argument("--label_col", default="label", help="Label column (1=spam, 0=ham)")
    ap.add_argument("--model", default="linear_svc", choices=["linear_svc","logreg","nb"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out", default="models/spam_pipeline.joblib")
    ap.add_argument("--metrics_out", default="models/metrics.json")
    ap.add_argument("--meta_out", default="models/model_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    for c in text_cols + [args.label_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")

    X = (df[text_cols].fillna("").astype(str)).agg(" ".join, axis=1)
    y = df[args.label_col].astype(int).values  # 1 spam, 0 ham

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_SEED, stratify=y
    )

    pipe = build_pipeline(args.model)
    pipe.fit(X_tr, y_tr)

    # Probabilities for threshold tuning & curves
    probs = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None
    y_pred = (probs >= 0.5).astype(int) if probs is not None else pipe.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=["ham", "spam"], digits=4)
    cm = confusion_matrix(y_te, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "report": report,
    }

    # Threshold tuning (maximize F1 on validation)
    best_th = 0.5
    if probs is not None:
        ths = np.linspace(0.05, 0.95, 19)
        best = (best_th, -1.0)
        for t in ths:
            pred_t = (probs >= t).astype(int)
            f1 = f1_score(y_te, pred_t)
            if f1 > best[1]:
                best = (t, f1)
        best_th = float(best[0])
        metrics["best_threshold"] = best_th
        metrics["f1_at_best_threshold"] = float(best[1])
        # AUCs
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_te, probs))
            metrics["pr_auc"] = float(average_precision_score(y_te, probs))
        except Exception:
            pass

    # Save pipeline
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, args.out)

    # Save metrics
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save model meta
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "kaggle_csv",
        "csv_sha1": hashlib.sha1(open(args.csv, "rb").read()).hexdigest()[:12],
        "model": args.model,
        "vectorizer": {"type": "tfidf", "word_ngrams": [1, 2]},
        "best_threshold": best_th,
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(report)
    print(f"Accuracy: {acc:.4f}")
    if probs is not None:
        print(f"Best threshold (by F1): {best_th:.2f}")
    print(f"Saved model to {args.out}")
    print(f"Saved metrics to {args.metrics_out}")
    print(f"Saved model meta to {args.meta_out}")

if __name__ == "__main__":
    main()
