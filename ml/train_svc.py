"""Fast tier: TF-IDF + calibrated LinearSVC.

Run:  python -m ml.train_svc

LinearSVC has no native probabilities, so we wrap it in CalibratedClassifierCV
(Platt scaling). Calibration matters: the cascade escalates to DistilBERT only
when this model's confidence is low, so those confidence scores must be
trustworthy, not arbitrary decision-function magnitudes.
"""
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml import config
from ml.data import load_split


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=50000,
            strip_accents="unicode",
        )),
        # class_weight balances the strong Normal/Depression majority against
        # the rare Stress / Personality-disorder classes.
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0),
            method="sigmoid",
            cv=3,
        )),
    ])


def main():
    train = load_split("train")
    # Augment with short conversational phrasings so short chat isn't OOD.
    from ml.augment import build_augmentation
    aug = build_augmentation()
    train = pd.concat([train[["text", "label"]], aug[["text", "label"]]],
                      ignore_index=True)
    print(f"Training on {len(train)} rows ({len(aug)} augmented short-form).")
    pipe = build_pipeline()

    t0 = time.perf_counter()
    pipe.fit(train["text"].tolist(), train["label"].tolist())
    print(f"Trained LinearSVC pipeline in {time.perf_counter() - t0:.1f}s "
          f"on {len(train)} examples.")

    joblib.dump(pipe, config.SVC_PATH)
    print(f"Saved -> {config.SVC_PATH}")


if __name__ == "__main__":
    main()
