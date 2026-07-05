"""Inference wrapper for the fine-tuned DistilBERT (accurate tier).

Loaded lazily so the app and the fast tier work even before DistilBERT has been
fine-tuned. Uses PyTorch (the environment's TensorFlow is broken, and Torch is
the more portable choice anyway).
"""
import os
import threading

# Force transformers onto the PyTorch backend (broken TF in this environment).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np

from ml import config

_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded = False


def available() -> bool:
    return os.path.isdir(config.BERT_DIR) and os.path.exists(
        os.path.join(config.BERT_DIR, "config.json")
    )


def _ensure_loaded():
    global _model, _tokenizer, _loaded
    if _loaded:
        return _model is not None
    with _lock:
        if _loaded:
            return _model is not None
        _loaded = True
        if not available():
            return False
        try:
            import torch  # noqa: F401
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            _tokenizer = AutoTokenizer.from_pretrained(config.BERT_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(
                config.BERT_DIR)
            _model.eval()
        except Exception as e:  # pragma: no cover
            print(f"DistilBERT load failed: {e}")
            _model = _tokenizer = None
    return _model is not None


def predict(text: str):
    """Return (label_name, confidence, prob_vector) or None if unavailable."""
    if not _ensure_loaded():
        return None
    import torch
    with torch.no_grad():
        enc = _tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=128, padding=True)
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    return config.ID_TO_CLASS[idx], float(probs[idx]), probs
