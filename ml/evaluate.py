"""Evaluation + architecture ablation harness.

Run:  python -m ml.evaluate

Answers the interview question "why not one model?" with numbers, and presents
the two objectives separately so they are compared fairly:

  TABLE 1  Routine 7-class classification (no safety override):
           SVC only vs DistilBERT only vs confidence-gated cascade.
           -> does the cascade keep transformer-level accuracy while paying
              transformer cost on only the hard minority of messages?

  TABLE 2  Crisis safety layer (binary suicidal-vs-rest):
           recall / precision / F1 at the tuned operating point.
           -> the recall-first safety thesis, with its false-alarm cost stated.

Outputs reports/metrics.json and reports/confusion_matrix.png.
"""
import json
import time

import joblib
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from ml import bert_infer, config
from ml.cascade import Cascade
from ml.data import load_split


def _latency(predict_one, texts, n=400):
    times = []
    for t in texts[:n]:
        s = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - s) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def evaluate_labelwise(name, y_true, y_pred, extra=None):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    report = classification_report(
        y_true, y_pred, labels=list(range(len(config.CLASSES))),
        target_names=config.CLASSES, output_dict=True, zero_division=0)
    out = {"name": name, "accuracy": round(acc, 4),
           "macro_f1": round(float(macro_f1), 4),
           "per_class_f1": {c: round(report[c]["f1-score"], 4)
                            for c in config.CLASSES}}
    if extra:
        out.update(extra)
    return out


def crisis_binary_metrics(y_true, crisis_scores, threshold):
    cid = config.CLASS_TO_ID[config.CRISIS_CLASS]
    yt = (np.asarray(y_true) == cid).astype(int)
    yp = (np.asarray(crisis_scores) >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "flagged_pct": round(float(yp.mean()) * 100, 1),
        "true_crisis_pct": round(float(yt.mean()) * 100, 1),
    }


def _confusion_png(y_true, y_pred, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred,
                              labels=list(range(len(config.CLASSES))))
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(config.CLASSES)))
        ax.set_yticks(range(len(config.CLASSES)))
        ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(config.CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Routine cascade confusion matrix (row-normalized)")
        for i in range(len(config.CLASSES)):
            for j in range(len(config.CLASSES)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black",
                        fontsize=8)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=130)
        print(f"Saved confusion matrix -> {path}")
    except Exception as e:
        print(f"(confusion matrix skipped: {e})")


def main():
    test = load_split("test")
    texts = test["text"].tolist()
    y_true = test["label"].tolist()

    svc = joblib.load(config.SVC_PATH)
    cascade = Cascade()
    have_bert = bert_infer.available()

    table1 = []

    # A. SVC only.
    svc_pred = svc.predict(texts).tolist()
    med, p95 = _latency(lambda t: svc.predict([t]), texts)
    table1.append(evaluate_labelwise(
        "SVC only (fast tier)", y_true, svc_pred,
        {"latency_ms_median": round(med, 3), "latency_ms_p95": round(p95, 3),
         "cost": "near-zero"}))

    # B. DistilBERT only.
    if have_bert:
        bert_pred = [config.CLASS_TO_ID[bert_infer.predict(t)[0]] for t in texts]
        med, p95 = _latency(lambda t: bert_infer.predict(t), texts, n=200)
        table1.append(evaluate_labelwise(
            "DistilBERT only (accurate tier)", y_true, bert_pred,
            {"latency_ms_median": round(med, 3),
             "latency_ms_p95": round(p95, 3), "cost": "high"}))
    else:
        print("DistilBERT not trained yet -- Table 1 will omit BERT rows. "
              "Run `python -m ml.train_bert`, then re-run evaluate.\n")

    # C. Confidence-gated cascade (routine task, NO safety override).
    casc_pred, tiers = [], []
    for t in texts:
        p = cascade.predict(t, use_accurate=have_bert, use_safety=False)
        casc_pred.append(config.CLASS_TO_ID[p.label])
        tiers.append(p.tier)
    med, p95 = _latency(
        lambda t: cascade.predict(t, use_accurate=have_bert, use_safety=False),
        texts)
    share = {k: round(tiers.count(k) / len(tiers), 4)
             for k in ("fast", "accurate")}
    table1.append(evaluate_labelwise(
        "Cascade (SVC->BERT, gated)", y_true, casc_pred,
        {"latency_ms_median": round(med, 3), "latency_ms_p95": round(p95, 3),
         "tier_share": share,
         "escalated_to_bert_pct": round(share["accurate"] * 100, 1),
         "cost": f"~{round(share['accurate']*100,1)}% of BERT-only cost"}))

    _confusion_png(y_true, casc_pred,
                   f"{config.REPORTS_DIR}/confusion_matrix.png")

    # TABLE 2. Crisis safety layer (binary).
    crisis_obj = joblib.load(config.CRISIS_PATH)
    crisis_scores = crisis_obj["pipeline"].predict_proba(texts)[:, 1]
    thr = crisis_obj.get("threshold", config.CRISIS_THRESHOLD)
    table2 = {
        "tuned_recall_first": crisis_binary_metrics(y_true, crisis_scores, thr),
        "f1_optimal_0.5": crisis_binary_metrics(y_true, crisis_scores, 0.5),
    }

    payload = {"dataset": config.HF_DATASET, "n_test": len(y_true),
               "classes": config.CLASSES,
               "table1_routine_multiclass": table1,
               "table2_crisis_safety_layer": table2}
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Pretty print.
    print("\n" + "=" * 82)
    print("TABLE 1  Routine 7-class classification (no safety override)")
    print("-" * 82)
    print(f"{'Architecture':<34}{'Acc':>7}{'MacroF1':>9}{'Med ms':>9}"
          f"{'p95 ms':>9}{'%->BERT':>9}")
    for r in table1:
        esc = r.get("escalated_to_bert_pct", "-")
        print(f"{r['name']:<34}{r['accuracy']:>7.3f}{r['macro_f1']:>9.3f}"
              f"{r['latency_ms_median']:>9.2f}{r['latency_ms_p95']:>9.2f}"
              f"{str(esc):>9}")
    print("=" * 82)
    print("TABLE 2  Crisis safety layer (binary suicidal-vs-rest)")
    print("-" * 82)
    t = table2["tuned_recall_first"]
    print(f"Recall-first  (thr={t['threshold']}):  recall={t['recall']}  "
          f"precision={t['precision']}  F1={t['f1']}  "
          f"flagged={t['flagged_pct']}% (true={t['true_crisis_pct']}%)")
    f = table2["f1_optimal_0.5"]
    print(f"F1-balanced   (thr=0.5):   recall={f['recall']}  "
          f"precision={f['precision']}  F1={f['f1']}  flagged={f['flagged_pct']}%")
    print("=" * 82)
    print(f"\nWrote {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
