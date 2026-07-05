"""Accurate tier: fine-tune DistilBERT for 7-class mental-health classification.

Run (full):      python -m ml.train_bert
Run (quick CPU): python -m ml.train_bert --sample 8000 --epochs 1

A plain PyTorch training loop (no HF Trainer) -- the environment's Trainer stack
is version-broken, and an explicit loop is more transparent for a write-up. Uses
inverse-frequency class weights in the loss to counter class imbalance, and
saves a self-contained model dir that ml/bert_infer.py loads at serve time.
"""
import argparse
import os
import time

# Broken TensorFlow in this environment -> force the PyTorch backend before
# transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from ml import config
from ml.data import load_split


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                             padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.enc.items()}, self.labels[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for batch, labels in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(1).cpu().tolist())
        gold.extend(labels.tolist())
    return (accuracy_score(gold, preds),
            f1_score(gold, preds, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="cap train rows (0 = full) for faster CPU runs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train, val = load_split("train"), load_split("val")
    # Short-form augmentation so the transformer also handles short chat (else
    # it drags low-confidence short messages back to the majority class).
    from ml.augment import build_augmentation
    aug = build_augmentation()
    train = pd.concat([train[["text", "label"]], aug[["text", "label"]]],
                      ignore_index=True)
    print(f"Added {len(aug)} augmented short-form rows -> {len(train)} total.")
    if args.sample:
        per = max(1, args.sample // len(config.CLASSES))
        train = train.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), per), random_state=config.SEED)
        ).reset_index(drop=True)
        print(f"Sampled train -> {len(train)} rows")

    tok = AutoTokenizer.from_pretrained(config.BERT_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_BASE_MODEL, num_labels=len(config.CLASSES),
        id2label=config.ID_TO_CLASS, label2id=config.CLASS_TO_ID).to(device)

    counts = train["label"].value_counts().sort_index()
    w = (len(train) / (len(config.CLASSES) * counts)).to_numpy()
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        TextDataset(train["text"], train["label"], tok),
        batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(
        TextDataset(val["text"], val["label"], tok), batch_size=32)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(0.1 * total_steps), total_steps)

    for epoch in range(args.epochs):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for step, (batch, labels) in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            optim.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss {running/step:.4f} "
                      f"({(time.perf_counter()-t0)/step:.2f}s/step)")
        acc, f1 = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}] val_acc={acc:.4f} val_macro_f1={f1:.4f}")

    model.save_pretrained(config.BERT_DIR)
    tok.save_pretrained(config.BERT_DIR)
    print(f"Saved DistilBERT -> {config.BERT_DIR}")


if __name__ == "__main__":
    main()
