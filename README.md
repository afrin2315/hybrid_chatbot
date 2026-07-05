---
title: Hybrid Mental Health Chatbot
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Evidence-driven hybrid NLP triage chatbot for mental health.
---

# Hybrid Mental-Health Triage Chatbot

A safety-aware mental-health support chatbot built around a **confidence-gated
model cascade with an always-on, recall-optimized crisis safety layer**. The
architecture is not a stylistic choice — it is **chosen by measurement**, and the
evaluation that justifies it lives in [`REPORT.md`](REPORT.md) and
[`reports/metrics.json`](reports/metrics.json).

> **Why this design, and why not just one model?** Because the two jobs have
> different objectives. Routine emotion classification should be *cheap and
> accurate*; crisis detection should *never miss a real crisis*. A single model
> forces one operating point onto both. The cascade lets each objective be tuned
> and measured independently — and the numbers show it keeps transformer-level
> accuracy while paying transformer cost on only a minority of messages.

## The core idea in one picture

```
user message
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ SAFETY LAYER  (always on)                                     │
│ high-recall crisis detector — tuned for recall, not accuracy  │
│ score ≥ threshold ──────────────► CRISIS response (override)  │
└─────────────────────────────────────────────────────────────┘
     │ (not crisis)
     ▼
┌─────────────────────────────────────────────────────────────┐
│ FAST TIER   calibrated TF-IDF + LinearSVC  (~1 ms, ~free)     │
│ confidence ≥ gate ──────────────► use this label             │
└─────────────────────────────────────────────────────────────┘
     │ (low confidence only)
     ▼
┌─────────────────────────────────────────────────────────────┐
│ ACCURATE TIER  fine-tuned DistilBERT  (only the hard cases)   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
RESPONSE LAYER — Gemini (if configured) or local templates,
                 grounded by the real classifier label
```

**No classification is ever hand-coded or mocked.** Every emotional tag comes
from a trained model; the response layer only decides *how* to reply.

## Design thesis: errors are asymmetric

Missing a crisis (false negative) can be catastrophic; a false crisis alarm just
shows a supportive message and helpline. So the safety layer is deliberately
tuned for **high recall** and accepts lower precision — a documented,
quantified trade-off (see Table 2 in `REPORT.md`). This asymmetry is the reason
the safety layer is a *separate* component rather than one of the seven classes,
and it is the central research contribution of the project.

## Dataset

- **Source:** *Sentiment Analysis for Mental Health* (aggregated corpus),
  ~52.7k labeled statements, mirrored on the Hugging Face Hub as
  [`btwitssayan/sentiment-analysis-for-mental-health`](https://huggingface.co/datasets/btwitssayan/sentiment-analysis-for-mental-health).
- **Classes (7):** Normal, Stress, Anxiety, Depression, Bipolar,
  Personality disorder, Suicidal.
- **Splits:** stratified 70 / 15 / 15 train / val / test, seeded and
  reproducible (`ml/data.py`). Class imbalance is real and handled with
  class-weighted training, not by discarding data.

## Reproduce the results

```bash
pip install -r requirements-train.txt

# Fast path (SVC + crisis + evaluation) — a couple of minutes on CPU:
python -m ml.pipeline

# Full path (also fine-tunes DistilBERT) — slower on CPU:
python -m ml.pipeline --bert
```

This writes `reports/metrics.json` and `reports/confusion_matrix.png`, and prints
the two evaluation tables. Individual steps:

| Step | Command | Output |
|------|---------|--------|
| Prepare data | `python -m ml.data` | `data/{train,val,test}.parquet` |
| Fast tier | `python -m ml.train_svc` | `artifacts/svc_pipeline.joblib` |
| Safety layer | `python -m ml.crisis` | `artifacts/crisis_detector.joblib` |
| Accurate tier | `python -m ml.train_bert` | `artifacts/distilbert/` |
| Evaluate / ablate | `python -m ml.evaluate` | `reports/metrics.json` |

## Results

See [`REPORT.md`](REPORT.md) for the full write-up. Headline on the held-out test
set (7,660 examples, 7 classes):

**Table 1 — routine classification (no safety override):**

| Architecture | Accuracy | Macro-F1 | Median latency | % routed to BERT |
|---|---|---|---|---|
| SVC only (fast tier) | 0.775 | 0.716 | **1.5 ms** | — |
| DistilBERT only (accurate tier) | 0.770 | 0.731 | 58.6 ms | — |
| **Cascade (SVC → BERT, gated)** | **0.791** | **0.748** | 3.5 ms | 42% |

The gated cascade **beats both standalone models** on accuracy and macro-F1
while running ~17× faster than DistilBERT-only, because it invokes the
transformer on only the ~42% of messages the fast model is unsure about. That is
the evidence-backed answer to *"why not one model?"*

**Table 2 — crisis safety layer (binary, suicidal-vs-rest):**

| Operating point | Recall | Precision | F1 | % flagged |
|---|---|---|---|---|
| Max-safety mode (τ 0.18) | **0.957** | 0.480 | 0.640 | 41.5% |
| **Balanced — deployed default (τ 0.5)** | 0.799 | 0.638 | 0.710 | 26.1% |

The detector *can* catch **95.7%** of crises at its max-recall point, but it is
only well-calibrated on training-like (long) text; on short chat it is noisy
(bare "I am sad" spikes, real distress can sit low). So in deployment the safety
decision is **a high-precision phrase lexicon OR a very-confident ML score
(≥ 0.90)** — the lexicon carries recall on explicit/concerning phrasing, and the
model contributes only when certain. The recall-first ML point stays available
via `CRISIS_THRESHOLD`. Choosing this deliberately, rather than blindly
maximizing a noisy score, is the mature call. See REPORT.md §4 for detail.

*(Reproduce with `python -m ml.evaluate`; raw numbers in `reports/metrics.json`,
confusion matrix in `reports/confusion_matrix.png`.)*

## Run the app locally

```bash
pip install -r requirements.txt      # serving deps only
# (train models first, or point at prebuilt artifacts/)
python hybrid_app.py                 # http://127.0.0.1:5000/
```

The app degrades gracefully: if the DistilBERT artifacts are absent it serves on
the fast + safety tiers alone; if `GEMINI_API_KEY` is unset it uses local
response templates.

## Transparency (no black box)

Every `/chat` response includes an `explain` block exposing the full basis of the
decision:

```json
{
  "reply": "...",
  "emotion": "Anxiety",
  "confidence": 0.76,
  "explain": {
    "decided_by": "fast",
    "crisis": false,
    "crisis_score": 0.03,
    "class_probabilities": { "Anxiety": 0.76, "Stress": 0.11, "Normal": 0.07 }
  }
}
```

`decided_by` tells you which tier made the call (`safety` / `fast` / `accurate`),
so the routing itself is inspectable.

## Project structure

```
ml/
  config.py       shared paths, class labels, thresholds
  data.py         download, clean, stratified split
  train_svc.py    calibrated TF-IDF + LinearSVC (fast tier)
  crisis.py       high-recall crisis detector (safety layer)
  train_bert.py   DistilBERT fine-tune, plain PyTorch loop (accurate tier)
  cascade.py      inference core: safety override + confidence-gated routing
  evaluate.py     two-table ablation + confusion matrix + latency
  pipeline.py     one-command reproduction
responder.py      response layer (Gemini or local templates)
hybrid_app.py     Flask app: auth, persistent history, /chat, transparency
tests/            pytest suite (routing, crisis override, auth, persistence)
templates/        landing page + login + chat UI
```

## Tech stack

- **Backend:** Flask, Flask-CORS, Gunicorn
- **ML/NLP:** scikit-learn (TF-IDF, LinearSVC, calibration, LogisticRegression),
  PyTorch + Transformers (DistilBERT)
- **Data/eval:** Hugging Face `datasets`, pandas, matplotlib
- **DB:** SQLite (users + persistent chat history)
- **Deploy:** Docker (CPU-only PyTorch), Hugging Face Spaces
- **Optional:** Gemini for response generation

## Environment variables

| Variable | Purpose |
|----------|---------|
| `FLASK_SECRET_KEY` | stable, secure sessions |
| `SESSION_COOKIE_SECURE` | `1` for HTTPS deployments |
| `SESSION_COOKIE_SAMESITE` | `None` for Hugging Face Space embedding |
| `NVIDIA_API_KEY` | optional; enables NVIDIA Nemotron responses (key from build.nvidia.com) |
| `NVIDIA_MODEL` | Nemotron model slug from the NVIDIA API catalog |
| `GEMINI_API_KEY` | optional; enables Gemini responses (used if no NVIDIA key) |
| `DISABLE_ACCURATE_TIER` | `1` to serve on fast + safety tiers only (skip DistilBERT) |
| `SVC_CONFIDENCE_GATE` | fast→accurate escalation threshold |
| `CRISIS_THRESHOLD` | safety-layer trigger (lower = more recall) |
| `DB_PATH`, `HOST`, `PORT` | standard overrides |

## Limitations & honesty

- The dataset is social-media text with noisy, self-reported labels; the model
  reflects that distribution and is **not** a clinical diagnostic tool.
- The crisis layer trades precision for recall by design — expect false alarms.
- This is a research/portfolio system, **not** a substitute for professional
  mental-health care.

## Disclaimer

For educational and research purposes only. Not a substitute for licensed
mental-health care, diagnosis, or emergency services. If you are in crisis,
contact your local emergency number or a crisis line (e.g. 988 in the U.S.).
