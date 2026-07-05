"""Safety tier: high-recall crisis (suicidal) detector.

Run:  python -m ml.crisis

This is the core of the design thesis. Errors are ASYMMETRIC: a missed crisis
(false negative) can be catastrophic, while a false alarm merely shows a support
message. So this binary detector is NOT optimized for accuracy -- it is tuned on
the validation set to a target recall, and the resulting threshold is saved.

It runs ALWAYS, in parallel with the routine cascade, and can override it.
"""
import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from ml import config
from ml.data import load_split

TARGET_RECALL = 0.95


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=3,
            max_features=50000, strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", C=4.0,
        )),
    ])


def _binary(df):
    return (df["label"] == config.CLASS_TO_ID[config.CRISIS_CLASS]).astype(int)


def choose_threshold(y_true, scores, target_recall=TARGET_RECALL):
    """Lowest-precision-loss threshold that still hits target recall."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    # thr has len-1 vs prec/rec; align.
    best_t, best_p = 0.5, -1.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if r >= target_recall and p > best_p:
            best_p, best_t = p, float(t)
    return best_t, best_p


def main():
    import pandas as pd
    from ml.augment import build_augmentation
    train, val = load_split("train"), load_split("val")
    aug = build_augmentation()
    train = pd.concat([train[["text", "label"]], aug[["text", "label"]]],
                      ignore_index=True)
    print(f"Training crisis detector on {len(train)} rows (+{len(aug)} short-form).")
    pipe = build_pipeline()
    pipe.fit(train["text"].tolist(), _binary(train).tolist())

    val_scores = pipe.predict_proba(val["text"].tolist())[:, 1]
    y_val = _binary(val).to_numpy()
    thr, prec_at_thr = choose_threshold(y_val, val_scores)

    achieved_recall = float(
        ((val_scores >= thr) & (y_val == 1)).sum() / max(1, (y_val == 1).sum())
    )
    print(f"Crisis detector: threshold={thr:.3f} "
          f"-> val recall={achieved_recall:.3f}, precision={prec_at_thr:.3f}")

    joblib.dump({"pipeline": pipe, "threshold": thr,
                 "target_recall": TARGET_RECALL}, config.CRISIS_PATH)
    print(f"Saved -> {config.CRISIS_PATH}")

    # Persist the chosen threshold so serving and config agree.
    meta_path = os.path.join(config.REPORTS_DIR, "crisis_threshold.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "val_recall": achieved_recall,
                   "val_precision": prec_at_thr,
                   "target_recall": TARGET_RECALL}, f, indent=2)


if __name__ == "__main__":
    main()
