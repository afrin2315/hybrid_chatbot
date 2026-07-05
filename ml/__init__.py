"""Real, reproducible ML pipeline for the hybrid mental-health triage system.

Modules:
    config    - shared paths, class labels, thresholds
    data      - dataset download, cleaning, stratified split
    train_svc - TF-IDF + calibrated LinearSVC (fast tier)
    train_bert- DistilBERT fine-tune (accurate tier, PyTorch)
    crisis    - high-recall crisis (suicidal) safety detector
    cascade   - confidence-gated cascade + safety override (inference core)
    evaluate  - metrics/latency/cost harness and architecture ablation
"""
