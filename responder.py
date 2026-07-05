"""Response layer: turns a classified emotional state into a supportive reply.

Separate from classification on purpose: the ML pipeline (ml/cascade) decides
*what* the user is feeling; this module decides *how* to respond. It uses Gemini
when a key is configured, grounded by the classifier's label, and otherwise
falls back to structured local templates that vary by message, turn, and the
last reply so the assistant never repeats itself verbatim.

Nothing here fabricates a classification -- the emotional tag always comes from
the real cascade.
"""
import os
import random

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                  f"{GEMINI_MODEL}:generateContent")

# NVIDIA Nemotron via NVIDIA's OpenAI-compatible API (key from build.nvidia.com).
# Set NVIDIA_MODEL to the exact slug shown in the NVIDIA API catalog for the
# model you have access to (e.g. an "...-nemotron-..." slug).
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL",
                              "nvidia/llama-3.1-nemotron-ultra-253b-v1")

CRISIS_REPLY = (
    "I'm really glad you reached out, and I want you to be safe. You don't have "
    "to go through this alone. If you might be in immediate danger or feel like "
    "you could hurt yourself, please contact your local emergency number now. "
    "In the U.S. you can call or text 988 (Suicide & Crisis Lifeline), any time. "
    "Are you safe right now, and is there someone nearby you can reach out to?"
)

# For each emotional class: several openers / coping steps / questions. We pick
# from each list deterministically per (label, turn, message) so the reply is
# stable on retry but varies across turns and messages.
_STRATEGY = {
    "Depression": {
        "validate": [
            "I'm sorry you're feeling low - that heaviness is real and it's okay to name it.",
            "That sounds genuinely disappointing, and it makes sense it's weighing on you.",
            "Feeling down about that is completely understandable.",
        ],
        "step": [
            "If it helps, we could look at one small, doable thing for today.",
            "We can keep this gentle - what's one tiny step that might ease the next hour?",
            "Sometimes naming the hardest part out loud takes a little of its weight.",
        ],
        "ask": [
            "What part of it is hurting the most right now?",
            "How long have you been carrying this feeling?",
            "What would feel even slightly supportive at this moment?",
        ],
    },
    "Anxiety": {
        "validate": [
            "Anxiety can make everything feel urgent and loud at once.",
            "That worry sounds exhausting to sit with.",
            "It's understandable that this is making you feel on edge.",
        ],
        "step": [
            "If you can, try 3 slow breaths - in for 4, out for 6.",
            "A quick grounding can help: name 5 things you see, 4 you can feel.",
            "Unclenching your jaw and shoulders for one breath can take the edge off.",
        ],
        "ask": [
            "What's the main worry looping in your mind right now?",
            "When did you first notice it getting heavier today?",
            "What's the very next thing you feel you have to face?",
        ],
    },
    "Stress": {
        "validate": [
            "That sounds like a lot to carry at once.",
            "It makes sense you're feeling stretched thin.",
            "Being under that much pressure is genuinely draining.",
        ],
        "step": [
            "It can help to pick just the next single step instead of the whole pile.",
            "We could sort this into 'now' vs 'later' if that would lighten it.",
            "Even a two-minute pause can reset things a little.",
        ],
        "ask": [
            "What's putting the most pressure on you today?",
            "If one thing got easier, which would help the most?",
            "What's due soonest that's on your mind?",
        ],
    },
    "Bipolar": {
        "validate": [
            "Shifts in mood and energy can be really disorienting.",
            "That up-and-down can be hard to ride out.",
        ],
        "step": [
            "Noticing where you are today - higher, lower, or mixed - can help.",
            "We can go at whatever pace feels manageable right now.",
        ],
        "ask": [
            "How would you describe your energy over the last few days?",
            "What does today feel like for you so far?",
        ],
    },
    "Personality disorder": {
        "validate": [
            "Strong emotions and relationships can feel overwhelming - that's heavy.",
            "It's a lot to hold when feelings run this intense.",
        ],
        "step": [
            "We can slow this down and take one piece at a time.",
            "There's no rush - we can stay with just this moment.",
        ],
        "ask": [
            "What feels most intense for you right now?",
            "What's the part you'd most like to be understood?",
        ],
    },
}

_SUGGESTIONS = {
    "Suicidal": ["I'm not safe right now", "I'm safe, but I need help",
                 "Help me find support"],
    "Depression": ["I feel empty", "I can't get motivated",
                   "One small step I can take"],
    "Anxiety": ["Help me calm down", "Break this into steps",
                "What should I do next?"],
    "Stress": ["Help me prioritize", "This is too much", "Make a plan with me"],
    "Bipolar": ["My mood keeps shifting", "I can't sleep", "I feel wired"],
    "Personality disorder": ["I feel overwhelmed", "It's about a relationship",
                             "Help me slow down"],
    "Normal": ["I want to vent", "Help me make a plan", "Ask me a question"],
}


def suggestions_for(label: str):
    return _SUGGESTIONS.get(label, _SUGGESTIONS["Normal"])


def _pick(options, seed="", avoid=""):
    """Randomized choice that won't echo `avoid` when alternatives exist.
    (Chat doesn't need deterministic replay; variety matters more.)"""
    if not options:
        return ""
    pool = [o for o in options if o.strip() != (avoid or "").strip()] or options
    return random.choice(pool)


def _last_bot_text(history):
    for m in reversed(history or []):
        if m.get("type") in ("model", "bot") and m.get("text"):
            return m["text"]
    return ""


def local_reply(label: str, user_message: str = "", history=None) -> str:
    msg = (user_message or "").strip()
    msg_l = msg.lower()
    turn = 1 + sum(1 for m in (history or []) if m.get("type") == "user")
    last = _last_bot_text(history)
    seed = f"{label}|{turn}|{msg_l}"

    # Emotional classes: validate + gentle step + open question.
    strat = _STRATEGY.get(label)
    if strat:
        parts = [
            _pick(strat["validate"], seed + "v"),
            _pick(strat["step"], seed + "s"),
            _pick(strat["ask"], seed + "a"),
        ]
        reply = " ".join(p for p in parts if p)
        if reply.strip() == last.strip():  # very defensive against repeats
            reply = _pick(strat["validate"], seed + "v2") + " " + \
                _pick(strat["ask"], seed + "a2")
        return reply

    # Normal / casual: detect intent so it isn't a canned line.
    if msg_l in ("hi", "hii", "hey", "hello", "hai", "yo", "hola") or \
            msg_l.startswith(("hi ", "hey ", "hello ")):
        return _pick([
            "Hi, glad you're here. How are you feeling today?",
            "Hey. I'm here to listen - what's on your mind right now?",
            "Hello. What would you like to talk about today?",
        ], seed, last)

    if "how are you" in msg_l or "how r u" in msg_l:
        return _pick([
            "Thanks for asking - I'm here with you. How are *you* feeling right now?",
            "I'm here and listening. How are you doing today, really?",
        ], seed, last)

    if msg_l in ("i want to vent", "let me vent", "vent"):
        return "Go ahead - I'm listening. What's been building up that you " \
               "haven't gotten to say out loud?"

    if "make a plan" in msg_l or msg_l in ("plan", "help me make a plan"):
        return ("Okay, let's make it simple.\n1) What's the goal?\n2) What's "
                "one small step you could take in the next 10 minutes?\n3) What "
                "might get in the way?\nWhat are we planning for?")

    if "ask me a question" in msg_l or msg_l == "question":
        return _pick([
            "What's been taking up most of your mental space lately?",
            "If today could go a little better, what would change first?",
            "When do you feel most like yourself?",
        ], seed, last)

    if msg_l in ("thanks", "thank you", "ty", "thx"):
        return _pick([
            "Anytime. I'm here whenever you want to talk more.",
            "Of course. Is there anything else on your mind?",
        ], seed, last)

    # Generic Normal: acknowledge the message, then an open question (varied),
    # regenerated if it would exactly repeat the last reply.
    acks = ["Thanks for sharing that.", "I hear you.",
            "Got it - thanks for telling me.", "That makes sense.",
            "I appreciate you telling me."]
    qs = ["What's on your mind about it right now?",
          "How are you feeling about that?",
          "Want to talk through what's been going on?",
          "What's been the hardest part of it?",
          "What would feel most helpful - talking it through, a next step, or just venting?"]
    for _ in range(4):
        candidate = f"{_pick(acks)} {_pick(qs)}"
        if candidate.strip() != last.strip():
            return candidate
    return candidate


def _persona(label: str) -> str:
    return (
        "You are a supportive, empathetic mental health coach. The user's "
        f"current state has been classified as '{label}' by a text classifier. "
        "Respond specifically and gently to this state and to what they actually "
        "said. Validate their feelings, offer non-clinical support, avoid "
        "diagnosing, and end with one open-ended question. Keep it concise "
        "(2-4 sentences). Do not repeat your previous message."
    )


def _to_openai_messages(label, history, user_message):
    msgs = [{"role": "system", "content": _persona(label)}]
    for m in (history or []):
        msgs.append({"role": "user" if m["type"] == "user" else "assistant",
                     "content": m["text"]})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _call_nvidia(user_message, label, history):
    """NVIDIA Nemotron via the OpenAI-compatible chat-completions endpoint."""
    r = requests.post(
        NVIDIA_API_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                 "Accept": "application/json"},
        json={"model": NVIDIA_MODEL,
              "messages": _to_openai_messages(label, history, user_message),
              "temperature": 0.7, "max_tokens": 240},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(user_message, label, history):
    formatted = [{"role": "user" if m["type"] == "user" else "model",
                  "parts": [{"text": m["text"]}]} for m in (history or [])]
    payload = {
        "contents": [{"role": "user", "parts": [{"text": _persona(label)}]},
                     *formatted,
                     {"role": "user", "parts": [{"text": user_message}]}],
        # Gemini 2.5 spends output tokens on internal "thinking"; disable it so
        # short replies aren't truncated, and keep a comfortable token budget.
        "generationConfig": {"temperature": 0.85, "maxOutputTokens": 400,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    r = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload,
                      timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("candidates") and j["candidates"][0].get("content"):
        return j["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise ValueError("empty Gemini response")


def generate(user_message: str, label: str, history, is_crisis: bool) -> str:
    """Empathetic reply grounded by the real classifier label.

    Provider order: NVIDIA Nemotron -> Gemini -> local templates. The crisis
    reply is always local and deterministic, so safety never depends on an
    external API being reachable.
    """
    if is_crisis or label == "Suicidal":
        return CRISIS_REPLY

    for name, key, fn in (("NVIDIA", NVIDIA_API_KEY, _call_nvidia),
                          ("Gemini", GEMINI_API_KEY, _call_gemini)):
        if not key:
            continue
        try:
            return fn(user_message, label, history)
        except Exception as e:
            print(f"{name} call failed ({e}); falling back.")

    return local_reply(label, user_message, history)
