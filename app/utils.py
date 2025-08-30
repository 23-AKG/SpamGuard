# app/utils.py  (drop-in)
import re
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from email import policy
from email.parser import BytesParser

def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def normalize_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def extract_text_from_pdf(file_stream) -> str:
    reader = PdfReader(file_stream)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(chunks)
    if "<html" in text.lower():
        text = clean_html(text)
    return normalize_ws(text)

def extract_text_from_eml(raw_bytes: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = msg.get("subject") or ""
    body = []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for p in parts:
        ct = p.get_content_type()
        if ct in ("text/plain", "text/html"):
            try:
                txt = p.get_content()
            except Exception:
                payload = p.get_payload(decode=True) or b""
                txt = payload.decode("utf-8", "ignore")
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", "ignore")
            if ct == "text/html":
                txt = clean_html(txt)
            body.append(txt)
    return normalize_ws(subject + "\n\n" + "\n".join(body))

def extract_any(file_storage):
    """Accept .pdf and .eml"""
    name = (file_storage.filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_storage.stream)
    data = file_storage.read()
    file_storage.seek(0)
    if name.endswith(".eml"):
        return extract_text_from_eml(data)
    raise ValueError("Unsupported file type (use .pdf or .eml)")

def top_contributors(model, text: str, k: int = 6):
    """
    Return top contributing features for spam score for linear models.
    """
    try:
        vec = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        X = vec.transform([text])
        if hasattr(clf, "classes_") and hasattr(clf, "predict_proba"):
            # If calibrated or logreg, use coef_ when available
            if hasattr(clf, "base_estimator") and hasattr(clf.base_estimator, "coef_"):
                coef = clf.base_estimator.coef_[0]
            elif hasattr(clf, "coef_"):
                coef = clf.coef_[0]
            else:
                return []
        elif hasattr(clf, "coef_"):
            coef = clf.coef_[0]
        else:
            return []
        import numpy as np
        contrib = (X.multiply(coef)).toarray()[0]
        idx = np.argsort(contrib)[-k:][::-1]
        feats = vec.get_feature_names_out()
        return [(feats[i], float(contrib[i])) for i in idx if contrib[i] > 0]
    except Exception:
        return []
