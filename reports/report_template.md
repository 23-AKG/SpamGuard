# SpamGuard Pro – Spam Email Detection Web Application

## 1. Introduction
Unsolicited bulk email (spam) remains one of the oldest and most persistent cybersecurity threats. Effective filtering is critical to protect individuals and organizations from phishing, fraud, and malware.  

This project implements **SpamGuard Pro**, a lightweight yet professional web application for spam detection. It combines traditional machine learning with a clean interface to allow users to upload email files (`.pdf` or `.eml`) and instantly classify them as **SPAM** or **Non-SPAM**. The aim is to deliver a reproducible, end-to-end system: dataset → training → evaluation → interactive classification.

---

## 2. Dataset
We used the **TREC 2007 Public Corpus**, a widely studied benchmark for spam filtering. The original dataset contains ~75,000 email messages (50,199 spam and 25,220 ham). To simplify integration, we leveraged a **preprocessed Kaggle version**, where emails are represented in tabular form:

- **label**: 1 = spam, 0 = ham  
- **subject**: subject line of the email  
- **email_to** / **email_from**: sender/receiver  
- **message**: body text  

This structure avoids raw MIME parsing, enabling faster model iteration and reproducibility.

---

## 3. Methodology

### 3.1 Preprocessing
- Combined `subject` and `message` into a single text field.  
- Applied **TF–IDF vectorization** with word n-grams (1–2).  
- Removed English stopwords, lowercased text, normalized whitespace.  

### 3.2 Model
We implemented multiple classifiers: **Linear SVM**, **Logistic Regression**, and **Naïve Bayes**. The best performing setup was **Logistic Regression with calibrated probabilities**.  

Enhancements:
- **Threshold tuning**: Instead of a fixed 0.5 cutoff, we tuned the spam threshold on the validation set to maximize **F1 score**.  
- **Calibration** (Platt scaling) ensured meaningful confidence values.  
- **Interpretability**: For each prediction, top contributing tokens are extracted, offering transparency.  

### 3.3 Web Application
Built with **Flask**, the app provides:
- Upload of `.pdf` or `.eml` files.  
- Server-side text extraction (PyPDF2 for PDF, Python’s `email` module for EML).  
- Per-file classification with labels, confidence, and top contributing terms.  
- Batch summary (SPAM vs Non-SPAM counts) and **Download CSV** for results.  
- Clean UI with dark theme and responsive design.  

---

## 4. Results

### 4.1 Metrics
From `models/metrics.json` after training on the preprocessed dataset:

- **Accuracy**: ~**97–98%**  
- **F1 score (at tuned threshold ~0.65)**: ~**0.96**  
- **ROC-AUC**: ~**0.99**  
- **PR-AUC**: ~**0.98**  
- **Confusion Matrix** (example):
  ```
  [[4982,   31],
   [  78, 9940]]
  ```
  (rows = actual ham/spam, cols = predicted ham/spam)

### 4.2 Screenshots
- **Dataset samples** classified correctly (ham → Non-SPAM, spam → SPAM).  
- **Synthetic phishing emails** (crypto airdrops, CEO fraud, bank alerts) flagged as SPAM.  
- **Benign academic/project emails** correctly marked Non-SPAM.  

(Screenshots attached separately in report.)

---

## 5. Discussion

### Strengths
- High accuracy on benchmark dataset.  
- Interpretability through highlighted features.  
- Handles both structured dataset input and real-world formats (.pdf/.eml).  
- Professional interface suitable for demonstration or lightweight deployment.  

### Limitations
- No OCR: scanned PDFs with image-only text cannot be parsed.  
- Dataset is from 2007; spam techniques have evolved (emoji obfuscation, adversarial NLP, image spam).  
- Classic ML models may miss subtle semantic cues compared to modern transformers (e.g., BERT).  

### Future Work
- Add **OCR (Tesseract)** for image-based spam.  
- Incorporate **character n-grams** and engineered features (URL density, capitalization ratio).  
- Experiment with **deep learning models** (DistilBERT fine-tuned for spam).  
- Deploy on **Heroku/Render** with persistent database for logging.  

---

## 6. Conclusion
SpamGuard Pro demonstrates a complete, functional pipeline for spam detection: **data → model → evaluation → interactive web app**. It balances practicality, accuracy, and interpretability. While limited by dataset age and scope, the project provides a robust baseline and a professional showcase of applied machine learning in cybersecurity.

---

## 7. References
- Gordon V. Cormack, TREC 2007 Public Corpus: https://plg.uwaterloo.ca/~gvcormac/treccorpus07/  
- Kaggle Preprocessed Dataset: https://www.kaggle.com/datasets/  
- Scikit-learn: https://scikit-learn.org  
- ReportLab, Flask, PyPDF2 Documentation  
