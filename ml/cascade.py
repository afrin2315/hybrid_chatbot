"""Inference core: confidence-gated cascade with an always-on safety override.

Decision flow for each message:

    1. SAFETY (always runs): the high-recall crisis detector scores the message.
       If score >= CRISIS_THRESHOLD -> return CRISIS immediately. This override
       is intentionally trigger-happy (recall over precision).

    2. FAST TIER: the calibrated LinearSVC classifies. If its top-class
       confidence >= SVC_CONFIDENCE_GATE, we trust it and stop here -- this is
       the cheap, sub-millisecond path that handles the easy majority.

    3. ACCURATE TIER: only for low-confidence messages do we pay for DistilBERT.
       If DistilBERT isn't available, we fall back to the SVC's best guess.

The whole point is measurable: `tier` in the result records which stage decided,
so evaluate.py can report how much traffic each tier actually handles.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import joblib

from ml import bert_infer, config

# High-precision explicit self-harm / suicidal-ideation phrases. The learned
# detector can miss paraphrases (e.g. tokenization drops the apostrophe in
# "don't"), so a curated lexicon runs alongside it and can only ever RAISE
# recall -- it never suppresses a crisis. This is a documented safety net, not a
# substitute for the model. Matching is apostrophe- and spacing-insensitive.
_CRISIS_PATTERNS = [
    r"kill (myself|me)", r"killing myself", r"end (my life|it all|myself)",
    r"take my (own )?life", r"want to die", r"wanna die", r"want to be dead",
    r"better off dead", r"dont want to (live|be here|be alive|exist|wake up)",
    r"no reason to (live|go on)", r"no point (in )?living", r"suicidal",
    r"suicide", r"hurt myself", r"harm myself", r"cut myself",
    r"cant go on( anymore)?", r"cant take it anymore", r"want to disappear",
    r"end my suffering", r"cant do this anymore",
    # concerning non-explicit ideation the ML detector scores unreliably
    r"no way out", r"cant see (a|any) way out", r"see no way out",
    r"life is pointless", r"whats the point of (living|life)",
    r"tired of living", r"give up on life", r"dont see the point anymore",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS))


def _explicit_crisis(text: str) -> bool:
    # Normalize: lowercase, strip apostrophes ("don't" -> "dont"), collapse ws.
    t = re.sub(r"\s+", " ", (text or "").lower().replace("'", "").replace("’", ""))
    return bool(_CRISIS_RE.search(t))


@dataclass
class Prediction:
    label: str
    confidence: float
    tier: str                      # "safety" | "fast" | "accurate"
    crisis: bool
    crisis_score: float
    probs: dict = field(default_factory=dict)


class Cascade:
    def __init__(self, svc_gate: float = None, crisis_threshold: float = None):
        self.svc = joblib.load(config.SVC_PATH)
        crisis_obj = joblib.load(config.CRISIS_PATH)
        self.crisis_pipe = crisis_obj["pipeline"]
        # The detector's stored threshold is the recall-first (max-safety) point;
        # we serve at the balanced operating point from config unless overridden.
        self.recall_first_threshold = crisis_obj.get("threshold", 0.18)
        self.crisis_threshold = (crisis_threshold if crisis_threshold is not None
                                 else config.CRISIS_THRESHOLD)
        self.svc_gate = (svc_gate if svc_gate is not None
                         else config.SVC_CONFIDENCE_GATE)

    def _svc_predict(self, text: str):
        probs = self.svc.predict_proba([text])[0]
        idx = int(probs.argmax())
        label = config.ID_TO_CLASS[idx]
        prob_map = {config.ID_TO_CLASS[i]: float(p) for i, p in enumerate(probs)}
        return label, float(probs[idx]), prob_map

    def predict(self, text: str, use_accurate: bool = True,
                use_safety: bool = True) -> Prediction:
        text = (text or "").strip()
        ml_score = float(self.crisis_pipe.predict_proba([text])[0, 1])
        # Hybrid safety score = learned detector OR explicit-phrase net.
        explicit = _explicit_crisis(text)
        crisis_score = max(ml_score, 0.99) if explicit else ml_score

        # 1. Safety override. Disabled (use_safety=False) only when measuring
        # the routine multiclass task in isolation, so the recall-first crisis
        # layer isn't mixed into the 7-class accuracy number.
        if use_safety and (explicit or crisis_score >= self.crisis_threshold):
            return Prediction(
                label=config.CRISIS_CLASS, confidence=crisis_score,
                tier="safety", crisis=True, crisis_score=crisis_score,
            )

        # 2. Fast tier.
        label, conf, probs = self._svc_predict(text)
        if conf >= self.svc_gate or not use_accurate:
            return Prediction(label=label, confidence=conf, tier="fast",
                              crisis=False, crisis_score=crisis_score,
                              probs=probs)

        # 3. Accurate tier (only for low-confidence messages).
        bert = bert_infer.predict(text) if use_accurate else None
        if bert is not None:
            blabel, bconf, bprobs = bert
            probs = {config.ID_TO_CLASS[i]: float(p)
                     for i, p in enumerate(bprobs)}
            return Prediction(label=blabel, confidence=bconf, tier="accurate",
                              crisis=(blabel == config.CRISIS_CLASS),
                              crisis_score=crisis_score, probs=probs)

        # Fallback: DistilBERT not trained yet -> keep the SVC guess.
        return Prediction(label=label, confidence=conf, tier="fast",
                          crisis=False, crisis_score=crisis_score, probs=probs)


_singleton: Optional[Cascade] = None


def get_cascade() -> Cascade:
    global _singleton
    if _singleton is None:
        _singleton = Cascade()
    return _singleton
