"""Dataset download, cleaning, and stratified train/val/test split.

Run once:  python -m ml.data

Produces data/{train,val,test}.parquet and data/label_map.json. The split is
seeded and stratified so every model in the ablation is trained and evaluated
on exactly the same partitions -- a prerequisite for a fair comparison.
"""
import json
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ml import config

_WS = re.compile(r"\s+")
_URL = re.compile(r"http\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Light, lossless-ish normalization. We deliberately keep casing/punctuation
    signals modest so both the linear model and the transformer see the same
    surface form."""
    if not isinstance(text, str):
        return ""
    t = _URL.sub(" ", text)
    t = t.replace("\r", " ").replace("\n", " ")
    t = _WS.sub(" ", t).strip()
    return t


def load_and_split():
    print(f"Downloading {config.HF_DATASET} ...")
    ds = load_dataset(config.HF_DATASET, split="train")
    df = ds.to_pandas()[[config.TEXT_COL, config.LABEL_COL]].copy()
    df.columns = ["text", "label_name"]

    # Clean + drop empties/dupes/unknown labels.
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 3]
    df = df[df["label_name"].isin(config.CLASSES)]
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after dedup; classes present:",
          sorted(df["label_name"].unique()))

    df["label"] = df["label_name"].map(config.CLASS_TO_ID)

    # Stratified: test first, then val out of the remainder.
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED,
        stratify=df["label"],
    )
    val_frac = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=config.SEED,
        stratify=train_val["label"],
    )

    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name}: {len(part)} rows")

    train.to_parquet(config.TRAIN_PATH, index=False)
    val.to_parquet(config.VAL_PATH, index=False)
    test.to_parquet(config.TEST_PATH, index=False)

    with open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": config.CLASSES,
                "class_to_id": config.CLASS_TO_ID,
                "crisis_class": config.CRISIS_CLASS,
                "counts": df["label_name"].value_counts().to_dict(),
            },
            f, indent=2,
        )
    print(f"Wrote splits to {config.DATA_DIR}")


def load_split(name: str) -> pd.DataFrame:
    path = {"train": config.TRAIN_PATH, "val": config.VAL_PATH,
            "test": config.TEST_PATH}[name]
    return pd.read_parquet(path)


if __name__ == "__main__":
    load_and_split()
