"""Shared configuration: paths, class labels, and routing thresholds.

Everything the pipeline needs to be reproducible lives here so training,
evaluation, and serving all agree on the same contract.
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

for _d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Dataset ---
# "Sentiment Analysis for Mental Health" (aggregated corpus), mirrored on the
# HF Hub. 52,681 rows, 7 clinical statuses.
HF_DATASET = "btwitssayan/sentiment-analysis-for-mental-health"
TEXT_COL = "statement"
LABEL_COL = "status"

# Canonical class order. Index == training label id. Do not reorder without
# retraining — serving relies on this mapping.
CLASSES = [
    "Normal",
    "Stress",
    "Anxiety",
    "Depression",
    "Bipolar",
    "Personality disorder",
    "Suicidal",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# The class that triggers the safety pathway. Missing this (false negative) is
# the costly error, so its detector is tuned for RECALL, not accuracy.
CRISIS_CLASS = "Suicidal"

# --- Split ---
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining (post-test) data

# --- Cascade routing ---
# Stage 1 (LinearSVC) decides alone when its calibrated confidence for the top
# class is >= this margin; otherwise the message escalates to DistilBERT.
# This value is chosen on the validation set by evaluate.py, not by hand.
SVC_CONFIDENCE_GATE = float(os.environ.get("SVC_CONFIDENCE_GATE", "0.70"))

# Crisis detector OPERATING threshold used when serving. The learned detector is
# well-calibrated on the (long-post) training distribution -- see REPORT.md
# Table 2 -- but noisy on very short chat text (e.g. bare "I am sad" spikes to
# ~0.76 while "I feel hopeless and empty" sits near 0.09). Rather than chase a
# threshold through that noise, the safety layer leans on a high-precision
# phrase lexicon (see ml/cascade.py) and only lets the ML score override when it
# is VERY confident (>= 0.90). Real explicit crisis text scores 0.95+, so it is
# still caught by the model; the lexicon backstops paraphrases the model misses.
CRISIS_THRESHOLD = float(os.environ.get("CRISIS_THRESHOLD", "0.90"))

# --- Artifact paths ---
SVC_PATH = os.path.join(ARTIFACTS_DIR, "svc_pipeline.joblib")
CRISIS_PATH = os.path.join(ARTIFACTS_DIR, "crisis_detector.joblib")
BERT_DIR = os.path.join(ARTIFACTS_DIR, "distilbert")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

BERT_BASE_MODEL = "distilbert-base-uncased"
