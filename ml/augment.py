"""Short-form data augmentation to reduce train/serve distribution mismatch.

The base corpus is long social-media posts, but users type short chat lines
("I am sad", "I didn't get the job"). Those are out-of-distribution, so the
models default to the majority class (Normal). We augment the TRAINING split
only with short, first-person, conversational phrasings for each class -- the
held-out validation/test splits are never touched, so reported metrics stay
honest. This is standard domain-adaptation practice, documented in REPORT.md.

Run standalone to inspect counts:  python -m ml.augment
"""
import pandas as pd

from ml import config

# Topic slots for {t}-templates -> everyday situations users actually mention.
_TOPICS = [
    "my job", "work", "the exam", "my grades", "money", "rent", "bills",
    "my relationship", "my breakup", "the interview", "a job offer",
    "my health", "the future", "my family", "my project", "school",
    "my career", "being alone", "my friend", "everything",
]

# Per-class short templates. "{t}" is filled with each topic; plain strings are
# used as-is. Kept deliberately conversational and short.
_TEMPLATES = {
    "Depression": [
        "i am sad", "i feel so sad", "im really sad", "i feel down",
        "i feel really down today", "i feel empty inside", "i feel hopeless",
        "i have no motivation", "i feel worthless", "i feel numb",
        "nothing feels worth it", "im so depressed", "i cant enjoy anything",
        "i feel low", "everything feels pointless lately", "i just feel sad",
        "i feel like a failure", "im sad about {t}", "i feel hopeless about {t}",
        "im really down about {t}", "i didnt get {t} and im devastated",
        "losing {t} broke me", "im heartbroken over {t}",
    ],
    "Anxiety": [
        "im so anxious", "i cant stop worrying", "my heart is racing",
        "i feel panicky", "im on edge", "i cant calm down", "im really nervous",
        "i feel anxious about {t}", "im so worried about {t}",
        "im scared about {t}", "im panicking about {t}", "im nervous about {t}",
        "i cant stop overthinking {t}", "i feel dread about {t}",
    ],
    "Stress": [
        "im so stressed", "im overwhelmed", "i cant cope", "im burnt out",
        "theres too much on my plate", "im under so much pressure",
        "im stressed about {t}", "im overwhelmed with {t}",
        "i cant cope with {t}", "{t} is stressing me out",
        "im drowning in {t}", "everything with {t} is piling up",
    ],
    "Bipolar": [
        "my mood keeps swinging", "i feel manic", "i cant sleep and feel wired",
        "my energy is all over the place", "i feel high then i crash",
        "my moods are so unstable lately", "i swing between highs and lows",
        "i havent slept and feel invincible", "i feel restless and racing",
    ],
    "Personality disorder": [
        "i feel empty and my relationships are chaotic", "i fear abandonment",
        "my emotions are so intense", "i dont know who i am",
        "i push people away then panic they'll leave",
        "my relationships always fall apart", "i feel everything so intensely",
        "i feel like i have no stable sense of self",
    ],
    "Suicidal": [
        "i want to die", "i dont want to be here anymore", "i want to end it all",
        "i cant go on anymore", "im thinking about ending my life",
        "i dont see the point in living", "id be better off dead",
        "i want to end my life", "i keep thinking about killing myself",
    ],
    "Normal": [
        "hi", "hii", "hello", "hey there", "how are you", "just checking in",
        "i had a good day", "i went for a walk", "i want to talk",
        "tell me about yourself", "what can you do", "im doing okay",
        "just curious", "what should we talk about", "how does this work",
        "i finished my work today", "i met a friend for coffee",
        "im feeling pretty good", "nothing much, just saying hi",
        "can you help me", "i had a productive day", "thanks for the chat",
    ],
}


def build_augmentation(max_per_class: int = 400, seed: int = config.SEED):
    rows = []
    for cls, templates in _TEMPLATES.items():
        variants = []
        for tpl in templates:
            if "{t}" in tpl:
                variants.extend(tpl.replace("{t}", t) for t in _TOPICS)
            else:
                variants.append(tpl)
        # de-dup, cap, seeded shuffle for reproducibility
        variants = list(dict.fromkeys(variants))
        df = pd.DataFrame({"text": variants})
        if len(df) > max_per_class:
            df = df.sample(max_per_class, random_state=seed)
        df["label_name"] = cls
        df["label"] = config.CLASS_TO_ID[cls]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[["text", "label_name", "label"]]


if __name__ == "__main__":
    aug = build_augmentation()
    print(f"Augmentation rows: {len(aug)}")
    print(aug["label_name"].value_counts().to_dict())
