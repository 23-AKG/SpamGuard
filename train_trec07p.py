import argparse, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from app.utils import clean_html, normalize_ws
from email import policy
from email.parser import BytesParser

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def parse_email_bytes(raw: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        subject = msg.get('subject') or ''
        parts = []
        if msg.is_multipart():
            for part in msg.iter_parts():
                ct = part.get_content_type()
                if ct in ['text/plain','text/html']:
                    try:
                        content = part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True) or b''
                        content = payload.decode('utf-8', errors='replace')
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    if ct == 'text/html':
                        content = clean_html(content)
                    parts.append(content)
        else:
            ct = msg.get_content_type()
            try:
                content = msg.get_content()
            except Exception:
                payload = msg.get_payload(decode=True) or b''
                content = payload.decode('utf-8', errors='replace')
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            if ct == 'text/html':
                content = clean_html(content)
            parts.append(content)
        text = f"{subject}\n\n" + "\n".join(parts)
        return normalize_ws(text)
    except Exception:
        return ""

def read_index(index_path: Path):
    items = []
    with open(index_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            label_str, rel = line.split()[:2]
            label = 1 if 'spam' in label_str.lower() else 0
            items.append((index_path.parent / rel, label))
    return items

def load_trec07p(root_dir: str, subset: int=None):
    root = Path(root_dir)
    candidates = [root / "full" / "index", root / "index", root / "trec07p" / "full" / "index"]
    items = []
    for c in candidates:
        if c.exists():
            items = read_index(c)
            break
    if not items: raise FileNotFoundError("index file not found")
    if subset:
        random.shuffle(items); items = items[:subset]
    texts, labels = [], []
    for p, y in items:
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        texts.append(parse_email_bytes(raw))
        labels.append(y)
    return texts, labels

def build_pipeline(model="linear_svc"):
    tfidf = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode', max_df=0.98, min_df=2, ngram_range=(1,2))
    if model=='linear_svc':
        clf=LinearSVC()
    elif model=='logreg':
        clf=LogisticRegression(max_iter=300)
    else:
        clf=MultinomialNB()
    return Pipeline([('tfidf', tfidf), ('clf', clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model', default='linear_svc', choices=['linear_svc','logreg','nb'])
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--out', default='models/spam_pipeline.joblib')
    ap.add_argument('--metrics_out', default='models/metrics.json')
    args = ap.parse_args()

    X, y = load_trec07p(args.data_dir, args.subset)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_SEED, stratify=y)
    pipe = build_pipeline(args.model)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=['ham','spam'], digits=4)
    cm = confusion_matrix(y_te, y_pred).tolist()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, args.out)
    from pathlib import Path
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.metrics_out,'w',encoding='utf-8') as f:
        json.dump({'accuracy':float(acc),'confusion_matrix':cm,'report':report}, f, indent=2)
    print(report); print(f"Accuracy: {acc:.4f}")
    print(f"Saved model to {args.out}"); print(f"Saved metrics to {args.metrics_out}")

if __name__=='__main__':
    main()
