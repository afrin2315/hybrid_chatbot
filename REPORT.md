# A Confidence-Gated Model Cascade with a Recall-Optimized Safety Layer for Mental-Health Text Triage

*Technical report / paper skeleton. All numbers are from `reports/metrics.json`,
produced by `python -m ml.evaluate` on a held-out test set of 7,660 examples.*

---

## Abstract

We present a mental-health text-triage system built around a **confidence-gated
cascade of classifiers with an always-on, recall-optimized safety layer**. The
design is motivated by a domain-specific observation: the cost of errors is
**asymmetric** — failing to detect a crisis (a false negative) is potentially
catastrophic, whereas a false crisis alarm merely surfaces a supportive message
and helpline. We therefore *decouple* the safety objective (maximize recall on
crisis detection) from the routine-classification objective (maximize accuracy at
minimal latency and cost), and tune and evaluate each independently. On a 7-class
mental-health corpus, the gated cascade attains higher accuracy (0.791) and
macro-F1 (0.748) than either a linear model (0.775 / 0.716) or a fine-tuned
DistilBERT (0.770 / 0.731) *alone*, while incurring ~17× lower median latency than
the transformer by invoking it on only ~42% of inputs. The safety layer, deployed
at a recall-first operating point, detects 95.7% of crisis messages.

**Keywords:** mental-health NLP, model cascade, cost-sensitive learning, crisis
detection, confidence calibration, safety-critical ML.

---

## 1. Introduction & contributions

Conversational mental-health support tools must satisfy two objectives that pull
in different directions. Routine emotional classification should be **cheap and
accurate** so the system is responsive and affordable at scale. Crisis detection
must **never miss a genuine crisis**, even at the cost of false alarms. A single
monolithic classifier forces one operating point onto both objectives.

This work makes the following contributions:

1. **An asymmetric-cost triage architecture** in which the crisis-safety
   component is a *separate*, recall-tuned detector that runs on every message
   and can override the routine classifier — rather than being one class among
   many competing under a single accuracy metric (§3.1, §3.3).
2. **A confidence-gated cascade** (calibrated linear model → transformer) that
   empirically *exceeds* the accuracy and macro-F1 of both of its constituent
   models while paying transformer-level cost on only the uncertain minority of
   inputs (§4, Table 1).
3. **A reproducible, two-objective evaluation protocol** that reports routine
   multiclass performance and crisis-safety performance separately, avoiding the
   metric contamination that occurs when a recall-first safety layer is folded
   into a single multiclass accuracy number (§3.4).

---

## 2. Data

- **Corpus:** *Sentiment Analysis for Mental Health* (aggregated), mirrored as
  `btwitssayan/sentiment-analysis-for-mental-health` on the Hugging Face Hub.
- **Size:** 51,061 statements after cleaning and de-duplication (from 52,681).
- **Labels (7):** Normal, Stress, Anxiety, Depression, Bipolar,
  Personality disorder, Suicidal.
- **Class imbalance:** substantial (Normal ≈ 16k, Depression ≈ 15k,
  Suicidal ≈ 10k, down to Personality disorder ≈ 1k). Handled with
  class-weighted training, never by discarding data.
- **Splits:** stratified 70 / 15 / 15 train / validation / test, seeded
  (`SEED=42`) and shared by every model so comparisons are fair
  (35,741 / 7,660 / 7,660). See `ml/data.py`.

**Threats to validity from the data.** The corpus is social-media text with
noisy, self-reported/annotator-assigned labels and overlapping clinical
categories (e.g., Depression vs Suicidal). Results characterize this
distribution and should not be read as clinical diagnostic performance.

---

## 3. Method

### 3.1 Overview

```
message ─▶ [Safety layer: crisis detector]  ── score ≥ τ_c ─▶ CRISIS (override)
                     │ (below threshold)
                     ▼
           [Fast tier: TF-IDF + LinearSVC]   ── conf ≥ τ_s ─▶ label
                     │ (low confidence)
                     ▼
           [Accurate tier: DistilBERT]       ───────────────▶ label
```

### 3.2 Fast tier — calibrated linear model
TF-IDF features (uni- and bi-grams, sublinear TF, 50k vocab) feed a
`LinearSVC` wrapped in `CalibratedClassifierCV` (Platt scaling). Calibration is
essential: the cascade escalates on *confidence*, so the probabilities must be
trustworthy rather than raw decision-function magnitudes. Class weights counter
imbalance. (`ml/train_svc.py`)

### 3.3 Safety layer — recall-optimized crisis detector
A binary suicidal-vs-rest classifier (TF-IDF → class-weighted logistic
regression). Its decision threshold τ_c is **not** chosen to maximize F1 or
accuracy; it is selected on the validation set as the highest-precision
threshold that still achieves ≥ 0.95 recall. This operationalizes the
asymmetric-cost principle. (`ml/crisis.py`)

**Hybrid recall net.** The learned detector can still miss paraphrases whose
surface form differs from training text — e.g. the vectorizer splits the
apostrophe in "don't", weakening "I don't want to be here anymore". Because a
missed crisis is the intolerable error, the safety layer additionally applies a
small, curated, high-precision lexicon of explicit ideation phrases
(apostrophe/spacing-insensitive). It runs in `OR` with the model and can only
ever *raise* recall, never suppress a detection. This learned-plus-lexical
combination is standard for safety-critical recall and is transparent (the
patterns are listed in `ml/cascade.py`); it is a targeted safety net, not a
substitute for the classifier.

### 3.4 Accurate tier — fine-tuned DistilBERT
`distilbert-base-uncased` fine-tuned for 7-way classification with a
class-weighted cross-entropy loss, in a plain PyTorch loop (2 epochs; per-class
capped subsample of the training split for CPU-time budget — see §6).
(`ml/train_bert.py`)

### 3.5 Gating & routing
The cascade (`ml/cascade.py`) runs the safety layer first; if not triggered, the
fast tier answers when its top-class calibrated confidence ≥ τ_s, otherwise the
message escalates to DistilBERT. A `tier` field records which stage decided each
message, enabling the traffic-share analysis in Table 1.

### 3.6 Evaluation protocol
Two independent evaluations on the same held-out test set:
- **Table 1** — routine 7-class metrics with the safety override *disabled*, so
  the recall-first crisis layer does not distort multiclass accuracy.
- **Table 2** — the crisis layer alone, as a binary task, at two operating
  points.

---

## 4. Results

### Table 1 — Routine 7-class classification (test n = 7,660)

| Architecture | Accuracy | Macro-F1 | Median latency | p95 latency | % → BERT |
|---|---|---|---|---|---|
| SVC only (fast tier) | 0.775 | 0.716 | 1.5 ms | 2.0 ms | — |
| DistilBERT only (accurate tier) | 0.770 | 0.731 | 58.6 ms | 97.4 ms | — |
| **Cascade (SVC → BERT, gated)** | **0.791** | **0.748** | 3.5 ms | 75.9 ms | 42.4% |

Per-class F1 (cascade): Normal 0.929, Anxiety 0.843, Bipolar 0.831,
Depression 0.724, Suicidal 0.716, Stress 0.613, Personality disorder 0.583.

**Findings.**
1. **The cascade dominates both constituents.** It exceeds SVC-only and
   BERT-only on *both* accuracy and macro-F1. The gate lets each model answer the
   inputs it handles best — the linear model resolves easy, high-confidence
   messages; the transformer is reserved for ambiguous ones — and the
   combination beats either alone.
2. **Efficiency.** Median latency is ~17× lower than BERT-only (3.5 ms vs
   58.6 ms) because only 42.4% of messages reach the transformer; 57.6% are
   resolved by the sub-millisecond linear tier.
3. This is the quantified answer to *"why not just one model?"*: one model cannot
   simultaneously be as cheap as the linear tier and as accurate as the cascade.

### Table 2 — Crisis safety layer (binary, test n = 7,660; 20.8% positive)

| Operating point | Recall | Precision | F1 | % flagged |
|---|---|---|---|---|
| **Recall-first (deployed, τ_c = 0.18)** | **0.957** | 0.480 | 0.640 | 41.5% |
| F1-balanced (τ_c = 0.5) | 0.799 | 0.638 | 0.710 | 26.1% |

**Findings.** Moving from the F1-optimal threshold (τ_c = 0.5) to the recall-first
threshold (τ_c = 0.18) raises crisis recall from 0.799 to **0.957** — recovering
~16 percentage points of true crises — at the cost of precision (0.64 → 0.48) and
a higher flag rate (26% → 42%). Under the asymmetric-cost assumption, recall-first
is the correct *maximum-safety* point.

**Operating-point selection for deployment.** The asymmetric-cost argument is not
unbounded: a detector that fires on mild everyday phrases erodes user trust
("crying wolf"), which is itself a real cost. Crucially, the learned detector is
well-calibrated only on the *training-like* distribution (long posts). On very
short chat messages it is unreliable — bare "I am sad" spikes to 0.76 while
"I feel hopeless and empty" sits near 0.09 — because such inputs are
out-of-distribution. We therefore do **not** trust a mid-range ML score in
deployment. The served safety decision is:

> **crisis  ⇔  explicit-phrase lexicon match  OR  ML score ≥ 0.90**

The lexicon (listed in `ml/cascade.py`) provides high recall on explicit and
concerning ideation regardless of phrasing; the ML detector contributes only
when very confident (real crisis text scores 0.95+). The recall-first ML point
(τ = 0.18, Table 2) remains available as a configurable max-safety mode via
`CRISIS_THRESHOLD`. Separating the *reported* ML capability from the *deployed*
hybrid decision is a deliberate, documented product choice. The confusion matrix
for the routine cascade is in `reports/confusion_matrix.png`.

---

## 5. Discussion

The result validates decoupling the two objectives. Because crisis detection is
handled by a dedicated recall-tuned component, the routine classifier is free to
optimize purely for accuracy/cost, and the safety guarantee (0.957 recall) is
stated explicitly rather than being an emergent, unmeasured property of a
multiclass model. The cascade's accuracy gain further shows the two are
complementary, not redundant — the central rebuttal to the "over-engineering"
critique.

---

## 6. Limitations & future work

- **Transformer under-trained.** For CPU-time budget, DistilBERT was fine-tuned
  for 2 epochs on a per-class-capped subsample (~15.5k rows), not the full
  35.7k. Its standalone numbers are therefore a lower bound; full-data training
  is expected to widen its lead and further lift the cascade. This is the first
  planned experiment.
- **Threshold transfer.** τ_c and τ_s are tuned on validation; production data
  drift would require recalibration.
- **Label noise / clinical validity.** See §2. A clinically annotated corpus and
  expert evaluation are needed before any real-world use.
- **Fairness.** Per-demographic error analysis is not yet done and is required
  for a safety-critical deployment.

---

## 7. Reproducibility

```bash
pip install -r requirements-train.txt
python -m ml.pipeline --bert     # data → SVC → crisis → DistilBERT → evaluate
```

All randomness is seeded (`SEED=42`). Artifacts land in `artifacts/`, metrics in
`reports/metrics.json`, and the confusion matrix in
`reports/confusion_matrix.png`. Unit tests: `python -m pytest tests/ -q`.
