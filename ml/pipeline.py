"""One-command reproduction of the whole ML pipeline.

    python -m ml.pipeline            # data -> svc -> crisis -> evaluate
    python -m ml.pipeline --bert     # also fine-tune DistilBERT (slow on CPU)

Every step is deterministic (fixed SEED in ml/config.py), so results reproduce.
"""
import argparse
import subprocess
import sys


def run(module, *args):
    print(f"\n{'='*70}\n>> python -m {module} {' '.join(args)}\n{'='*70}")
    r = subprocess.run([sys.executable, "-m", module, *args])
    if r.returncode != 0:
        sys.exit(f"Step failed: {module}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", action="store_true",
                    help="also fine-tune DistilBERT (accurate tier)")
    ap.add_argument("--bert-sample", type=int, default=21000)
    ap.add_argument("--bert-epochs", type=int, default=2)
    args = ap.parse_args()

    run("ml.data")
    run("ml.train_svc")
    run("ml.crisis")
    if args.bert:
        run("ml.train_bert", "--sample", str(args.bert_sample),
            "--epochs", str(args.bert_epochs))
    run("ml.evaluate")


if __name__ == "__main__":
    main()
