# 📧 SpamGuard Pro  
**Spam Email Detection Web Application**

SpamGuard Pro is a lightweight yet professional spam detection system.  
It combines machine learning with a clean web interface to classify emails as **SPAM** or **Non-SPAM**.  

Built with **Python, Flask, and scikit-learn**.  

---

## ✨ Features
- Train models on the **TREC 2007 Public Corpus** (preprocessed Kaggle version).  
- Upload `.pdf` or `.eml` files for classification.  
- Extracts text automatically and predicts with high accuracy.  
- Shows **confidence scores**, **decision threshold**, and **“why” tokens** (top contributing words).  
- Batch upload with **summary statistics** and **CSV download**.  
- Professional dark-theme UI.  

---

## 📊 Model Performance
- **Accuracy**: ~97–98%  
- **F1 Score (tuned threshold ~0.65)**: ~0.96  
- **ROC-AUC**: ~0.99  
- **PR-AUC**: ~0.98  

(See [`reports/SpamGuard_Report.md`](reports/SpamGuard_Report.md) for details.)  

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/spamguard_pro.git
cd spamguard_pro
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset
This project uses the **TREC 2007 Public Corpus** (preprocessed Kaggle version).  
- Kaggle: [Preprocessed TREC 2007 Public Corpus Dataset](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset)  
- Place the CSV under `data/processed_data.csv`  

---

## 🧠 Training a Model
Train on the CSV:
```bash
python train_csv.py --csv data/processed_data.csv --model logreg
```

Outputs:  
- `models/spam_pipeline.joblib` – trained model  
- `models/metrics.json` – evaluation metrics  
- `models/model_meta.json` – metadata  

---

## ▶️ Running the App
Start the Flask server:
```bash
python run.py
```

Visit **http://127.0.0.1:5000** in your browser.  

Upload `.pdf` or `.eml` emails and view predictions.  

---

## 🧪 Testing with Sample PDFs
You can auto-generate test emails (spam + ham + synthetic phishing):  
```bash
python make_test_pdfs.py --csv data/processed_data.csv --n_ham 3 --n_spam 3 --outdir test_pdfs
```
Then upload files from `test_pdfs/` in the app.  

---

## 📄 Reports
- Main project report: [`reports/report_template.md`](reports/report_template.md)
---

## 🚀 Deployment
Includes a **Procfile** for easy deployment to Heroku/Render.  

## Screen Shots
-(images/1.png)
-(images/2.png)
-(images/3.png)
-(images/4.png)
-(images/5.png)
-(images/6.png)
