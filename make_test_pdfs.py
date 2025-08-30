# make_test_pdfs.py
# Creates PDFs you can upload to SpamGuard Pro for quick manual testing.
# - Samples real rows from your processed_data.csv (spam + ham)
# - Generates several synthetic "crazy" emails (both spammy and legit)
#
# Usage:
#   python make_test_pdfs.py --csv data/processed_data.csv --n_ham 3 --n_spam 3 --outdir test_pdfs

import argparse, os, random, textwrap
import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def write_pdf(path, subject, message, sender="unknown@example.com", recipient="you@example.com"):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER
    left = 0.9 * inch
    top = height - 0.9 * inch
    line_height = 14

    def draw_line(y, text, bold=False):
        if bold:
            c.setFont("Helvetica-Bold", 11)
        else:
            c.setFont("Helvetica", 11)
        c.drawString(left, y, text)

    y = top
    draw_line(y, f"From: {sender}"); y -= line_height
    draw_line(y, f"To: {recipient}"); y -= line_height
    draw_line(y, f"Subject: {subject}", bold=True); y -= (line_height + 4)

    # wrap message
    c.setFont("Helvetica", 11)
    wrap_width = 85
    for para in message.split("\n"):
        for line in textwrap.wrap(para, width=wrap_width) or [""]:
            if y < 1.0 * inch:
                c.showPage()
                y = top
                c.setFont("Helvetica", 11)
            c.drawString(left, y, line)
            y -= line_height
        y -= 6  # paragraph spacing

    c.showPage()
    c.save()

def sample_from_csv(csv_path, n_ham=3, n_spam=3):
    df = pd.read_csv(csv_path)
    # Expecting: label (1 spam / 0 ham), subject, message, email_from, email_to
    # Fallback if columns differ:
    colmap = {c.lower(): c for c in df.columns}
    def pick(colname, default):
        return colmap.get(colname, default if default in df.columns else None)

    label_col = pick("label", "label")
    subj_col  = pick("subject", "subject")
    msg_col   = pick("message", "message")
    from_col  = pick("email_from", "email_from")
    to_col    = pick("email_to", "email_to")

    assert label_col in df.columns, "No 'label' column found (expected 0/1)."
    assert subj_col in df.columns and msg_col in df.columns, "CSV needs 'subject' and 'message' columns."

    df_spam = df[df[label_col] == 1].sample(min(n_spam, (df[label_col]==1).sum()), random_state=42)
    df_ham  = df[df[label_col] == 0].sample(min(n_ham,  (df[label_col]==0).sum()), random_state=42)

    out = []
    for i, row in enumerate(df_spam.itertuples(index=False), 1):
        out.append({
            "filename": f"dataset_spam_{i}.pdf",
            "subject":  getattr(row, subj_col, "No subject"),
            "message":  str(getattr(row, msg_col, ""))[:8000],
            "sender":   getattr(row, from_col, "unknown@example.com") if from_col else "unknown@example.com",
            "recipient":getattr(row, to_col,   "you@example.com")     if to_col else "you@example.com",
        })
    for i, row in enumerate(df_ham.itertuples(index=False), 1):
        out.append({
            "filename": f"dataset_ham_{i}.pdf",
            "subject":  getattr(row, subj_col, "No subject"),
            "message":  str(getattr(row, msg_col, ""))[:8000],
            "sender":   getattr(row, from_col, "colleague@example.com") if from_col else "colleague@example.com",
            "recipient":getattr(row, to_col,   "you@example.com")        if to_col else "you@example.com",
        })
    return out

def crazy_synthetics():
    # A few intentionally extreme/fishy and a few legit/neutral
    SYN = []

    SYN.append((
        "URGENT: Your Account Will Be Closed",
        """Dear Customer,

We detected suspicious login attempts from Lagos and Moscow.
To avoid permanent closure, verify your identity within 3 hours:
http://secure-verify-account.example.ru/click?token=987654321

Failure to act will result in immediate fund seizure.

Regards,
Security Team""",
        "security@bank-alerts.support", "you@example.com"
    ))

    SYN.append((
        "🔥 Congrats! You WON $10,000 in Crypto Airdrop",
        """CLAIM INSTANTLY: send a small verification payment of 0.02 BTC to:
bc1qth1s1sdefinitelyleg1t

Then reply with your wallet seed phrase to finalize the grant.
Offer expires in 30 minutes!""",
        "rewards@crypto-blaze.win", "you@example.com"
    ))

    SYN.append((
        "Invoice #38277 Overdue — Action Required",
        """Hi,

Our AR records show invoice #38277 (April) is 53 days overdue.
Please remit payment via the secure link:
https://pay.example.com/i/38277

Thanks,
Accounts""",
        "billing@vendor-payments.com", "you@example.com"
    ))

    SYN.append((
        "Team Standup Notes & Next Sprint Plan",
        """Folks,

Notes from today's standup are in the doc. Highlights:
- Roll out v2.3 to 10% traffic
- Fix flaky tests on CI
- Prepare metrics for the quarterly review

Regards,
PM""",
        "pm@company.com", "dev-team@company.com"
    ))

    SYN.append((
        "Professor Feedback on Your Draft",
        """Hi,

I left inline comments on sections 2 and 4. Please clarify the dataset split
and expand the error analysis. Good progress overall.

Best,
Prof. Rao""",
        "prof.rao@university.edu", "student@university.edu"
    ))

    SYN.append((
        "CEO: Wire 20,000 USD Now",
        """This is confidential. I need 20,000 USD wired within 30 minutes to close the deal.
Do NOT call me; I'm in a board meeting. Send to:
Account: 4455-XXXX
Routing: 021000021

— Sent from mobile""",
        "ceo.mobile@gmail.com", "finance@company.com"
    ))

    return [
        {
            "filename": f"synthetic_{i+1}.pdf",
            "subject": s, "message": m, "sender": fr, "recipient": to
        }
        for i, (s, m, fr, to) in enumerate(SYN)
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to processed_data.csv (Kaggle TREC07p preprocessed)")
    ap.add_argument("--n_ham", type=int, default=3)
    ap.add_argument("--n_spam", type=int, default=3)
    ap.add_argument("--outdir", default="test_pdfs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    jobs = sample_from_csv(args.csv, args.n_ham, args.n_spam) + crazy_synthetics()
    for j in jobs:
        outpath = os.path.join(args.outdir, j["filename"])
        write_pdf(outpath, j["subject"], j["message"], sender=j["sender"], recipient=j["recipient"])
    print(f"Created {len(jobs)} PDFs in: {args.outdir}")

if __name__ == "__main__":
    main()
