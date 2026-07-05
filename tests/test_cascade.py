"""Tests for the inference cascade and safety override.

Run:  python -m pytest tests/ -q
Requires the SVC + crisis artifacts (python -m ml.train_svc && python -m ml.crisis).
"""
import pytest

from ml import config
from ml.cascade import Cascade


@pytest.fixture(scope="module")
def cascade():
    return Cascade()


def test_clear_crisis_is_overridden(cascade):
    p = cascade.predict("I want to kill myself, I can't go on anymore")
    assert p.crisis is True
    assert p.label == config.CRISIS_CLASS
    assert p.tier == "safety"


def test_neutral_message_is_not_crisis(cascade):
    p = cascade.predict("had a good lunch and a walk in the park today")
    assert p.crisis is False
    assert p.crisis_score < cascade.crisis_threshold


def test_explicit_crisis_phrases_always_caught(cascade):
    # The lexical safety net must catch explicit ideation even when phrasing
    # (apostrophes, paraphrase) would otherwise slip past the learned detector.
    for msg in ["honestly I don't want to be here anymore",
                "I can't go on like this",
                "sometimes I think about killing myself",
                "I'd be better off dead"]:
        p = cascade.predict(msg)
        assert p.crisis is True, f"missed crisis: {msg}"
        assert p.tier == "safety"


def test_benign_low_mood_is_not_crisis(cascade):
    # The recall-first past-failure: mild sadness must NOT trigger a crisis.
    for msg in ["that really got me down today", "work is stressing me out"]:
        assert cascade.predict(msg).crisis is False, f"false alarm: {msg}"


def test_confident_message_stays_in_fast_tier(cascade):
    # A confident, clearly non-crisis message must be resolved by the fast tier
    # without escalating to the expensive transformer. Disable the safety layer
    # so we isolate the fast/accurate routing decision.
    p = cascade.predict("thanks so much, that was really helpful and kind",
                        use_safety=False)
    assert p.tier == "fast"
    assert p.label in config.CLASSES


def test_safety_can_be_disabled_for_measurement(cascade):
    # The routine-task evaluation path must never route through the safety tier.
    p = cascade.predict("I feel hopeless", use_safety=False)
    assert p.tier in ("fast", "accurate")


def test_prediction_confidence_in_range(cascade):
    p = cascade.predict("just feeling a bit tired today")
    assert 0.0 <= p.confidence <= 1.0
    assert 0.0 <= p.crisis_score <= 1.0
