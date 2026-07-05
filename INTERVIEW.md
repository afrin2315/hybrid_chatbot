# Interview Prep — Hybrid Mental-Health Triage Chatbot

Everything you need to defend this project confidently. Read the pitch and the
"Why not one model?" section until you can say them cold; skim the rest so
nothing surprises you. Numbers are from `reports/metrics.json` (test n = 7,660).

---

## 0. The 30-second pitch

> "It's a safety-aware mental-health support chatbot. The core is a
> **confidence-gated model cascade** — a fast linear classifier handles the easy
> majority of messages in about a millisecond, and only the uncertain ones
> escalate to a fine-tuned DistilBERT. Running on top of that is an **always-on,
> recall-tuned crisis-safety layer**, because the cost of missing a crisis is not
> symmetric with a false alarm. Every reply is fully explainable — you can see
> which model decided, its confidence, and the full class breakdown. I evaluated
> the whole thing on a 51k-example dataset and the cascade beats either model
> alone while being 17× cheaper than the transformer."

## 0b. The 2-minute version (add this)

> "The design is driven by one observation: routine emotion classification and
> crisis detection have *different objectives*. Routine classification should be
> cheap and accurate; crisis detection should never miss a real crisis, even at
> the cost of false alarms. A single model forces one operating point on both, so
> I split them. The routine path is a cascade: a calibrated TF-IDF + LinearSVC
> answers when it's confident, otherwise DistilBERT takes over. The safety path
> is a separate binary detector tuned for high recall, plus a curated phrase net
> for explicit ideation. I also built a two-table evaluation harness so the two
> objectives are measured independently, and a live transparency panel so nothing
> is a black box. It's deployed on Hugging Face as a Docker Space with auth and
> persistent history."

---

## 1. THE question: "Why three models? Why not just one?"

**Short answer:** "Because I measured it, and the combination wins."

**Full answer:**
> "First, it's really two classifiers plus a safety detector, each with a job.
> I ran an ablation on the same held-out test set:
> - LinearSVC alone: 0.775 accuracy, 0.716 macro-F1, ~1.5 ms
> - DistilBERT alone: 0.770 accuracy, 0.731 macro-F1, ~59 ms
> - **Cascade: 0.791 accuracy, 0.748 macro-F1, ~3.5 ms**
>
> The cascade beats *both* individual models on accuracy and macro-F1, and it's
> ~17× faster than DistilBERT because only ~42% of messages escalate to it. So a
> single model can't match it: the linear model alone is cheaper but less
> accurate, and the transformer alone is more expensive without being more
> accurate than the cascade. The gate sends each message to the model that
> handles it best."

**Follow-up: "Why does the cascade beat both?"**
> "The two models make different errors. The linear model is strong on clear,
> lexically obvious cases; the transformer helps on ambiguous ones. By only
> escalating low-confidence messages, I let each operate where it's strong — it's
> a confidence-based ensemble, not redundancy."

**Follow-up: "Isn't that over-engineering?"**
> "It would be if I couldn't justify it — that's exactly why I ran the ablation.
> If one model had won, I'd have shipped one model. The measurement *is* the
> justification. And the crisis layer is separate on purpose, which I can explain."

---

## 2. The safety layer / crisis detection (your strongest, most unique part)

**"Why is crisis detection separate from the 7-class model?"**
> "Because the error costs are asymmetric. Missing a real crisis is potentially
> catastrophic; a false crisis alarm just shows a supportive message and a
> helpline. If crisis were one of seven classes optimized for accuracy, the model
> would trade away recall to look good on the majority classes. So I made it a
> dedicated binary detector tuned for **recall**, not accuracy."

**"How is it tuned?"**
> "I select the threshold on the validation set as the highest-precision point
> that still hits ≥ 0.95 recall — a recall-first operating point. At that point it
> catches 95.7% of crises; at an F1-optimal threshold it'd catch only 79.9%. I
> knowingly accept lower precision to recover those ~16 points of real crises."

**"But doesn't that cause false alarms?"** (great question — you hit this yourself)
> "Yes, and I handled it. The learned detector is well-calibrated on the long-post
> training distribution but noisy on very short chat — 'I am sad' spiked while 'I
> feel hopeless and empty' scored low. So in deployment I don't trust a mid-range
> ML score: the safety decision is a **high-precision phrase lexicon OR a very
> confident ML score (≥ 0.90)**. The lexicon gives recall on explicit ideation
> regardless of phrasing; the model only overrides when it's sure. That killed the
> false alarms on mild sadness while still catching every explicit crisis."

**"Isn't a keyword list a hack? Didn't you say you removed keyword mocking?"**
> "Different thing. The old version *faked the whole classification* with keywords
> and hardcoded confidences. Here the classification is fully learned; the lexicon
> is a documented, transparent *recall net* layered on the learned detector for a
> safety-critical decision — which is standard practice. It can only raise recall,
> never suppress a detection, and the crisis reply is deterministic so safety
> never depends on an external API."

**Ethics angle (be ready):**
> "It's explicitly not a medical device — it triages and points to professional
> help (988), and the disclaimer is everywhere. The recall-first design reflects
> that in this domain a false negative is the dangerous error."

---

## 3. Data & ML questions

**"What dataset?"**
> "The aggregated 'Sentiment Analysis for Mental Health' corpus — ~51k statements
> after cleaning, 7 classes: Normal, Stress, Anxiety, Depression, Bipolar,
> Personality disorder, Suicidal. Stratified 70/15/15 split, seeded."

**"It's imbalanced — how did you handle it?"**
> "Real imbalance (Normal ~16k vs Personality disorder ~1k). I used class-weighted
> loss in every model rather than discarding data, and I report **macro-F1**, not
> just accuracy, so minority classes count equally."

**"Why calibrate the LinearSVC?"**
> "LinearSVC only gives decision-function margins, not probabilities. The cascade
> gate escalates on *confidence*, so I wrapped it in CalibratedClassifierCV (Platt
> scaling) to get trustworthy probabilities. An uncalibrated score would make the
> gate meaningless."

**"Why DistilBERT and not full BERT or an LLM?"**
> "DistilBERT is ~40% smaller and faster than BERT with ~97% of the performance —
> right for the 'accurate but still deployable' tier. A full LLM for
> classification would be far more expensive per call and overkill; I use an LLM
> only for the *response*, grounded by the classifier."

**"What about data augmentation — tell me about that."**
> "The base corpus is long Reddit-style posts, but users type short lines like 'I
> am sad'. Those are out-of-distribution, so the models defaulted to Normal. I
> augmented the **training split only** with ~450 short, conversational phrasings
> per-class, then retrained. The validation/test splits are untouched, so the
> reported metrics stay honest. It's standard domain adaptation, and it fixed the
> short-message behavior."

---

## 4. Evaluation & rigor (shows engineering maturity)

**"How did you evaluate it?"**
> "Two separate tables, because the two objectives shouldn't be mixed. Table 1 is
> routine 7-class classification with the safety override *off*, comparing SVC,
> DistilBERT, and the cascade on accuracy, macro-F1, per-class F1, and latency.
> Table 2 is the crisis layer as a binary task at two operating points. Mixing
> them would let the recall-first crisis layer wreck the multiclass accuracy and
> hide the real story."

**"How do you know you didn't overfit?"**
> "Held-out test set never used in training or threshold selection; thresholds are
> chosen on validation. Everything is seeded and reproducible with one command."

**"What metric matters most here?"**
> "Depends on the objective: macro-F1 for routine classification (handles
> imbalance), and **recall** for the crisis layer (asymmetric cost). Accuracy
> alone would be misleading on both."

---

## 5. Systems / engineering / scaling

**"How would this scale to millions of users?"**
> "The design already helps: ~58% of traffic never touches the transformer, so
> average cost/latency is dominated by the ~1 ms linear tier. To scale I'd batch
> DistilBERT inference, cache the model in a warm worker (I already use a lazy
> singleton), move sessions/history to Postgres + Redis, and put the transformer
> behind an autoscaling inference service. The cascade is essentially a cost
> optimizer — it's most valuable at scale."

**"Latency budget?"**
> "Median ~3.5 ms end-to-end for classification; p95 ~76 ms when a message
> escalates. The LLM response is the slow part, so I stream it and the safety
> reply is instant and local."

**"How is it deployed?"**
> "Docker on Hugging Face Spaces, gunicorn, CPU-only PyTorch to keep the image
> lean. Auth with hashed passwords + session cookies, persistent chat history in
> SQLite, and it degrades gracefully — if the transformer or the LLM key is
> missing, it still runs on the fast + safety tiers with local replies."

**"What's the response layer?"**
> "Provider-agnostic. It uses an LLM (Gemini, or NVIDIA Nemotron / OpenRouter)
> grounded by the classifier's label to write the reply, and falls back to
> structured local templates if no key is set. The LLM never does the
> classification or the safety decision — those stay in my models."

---

## 6. Transparency / explainability

**"How is it not a black box?"**
> "Every `/chat` response returns an `explain` block: which tier decided
> (safety/fast/accurate), the crisis score vs its threshold, and the top class
> probabilities. The UI renders that live — you literally watch the routing and
> the confidence for each message."

---

## 7. Limitations — say these BEFORE they ask (it builds trust)

- **Domain shift:** trained on long posts; short chat needed augmentation, and
  subtle/rare phrasings can still be misread. Real deployment needs in-domain,
  clinically-annotated data.
- **Label noise:** social-media labels are self-reported/annotator-assigned;
  categories overlap (Depression vs Suicidal). Not clinical ground truth.
- **CPU-time budget:** DistilBERT was fine-tuned on a per-class-capped subsample
  for a couple of epochs — its standalone number is a lower bound; full training
  would widen its lead and lift the cascade further.
- **Crisis precision:** recall-first by design, so false alarms happen; I mitigate
  with the operating-point choice + lexicon, but it's a real trade.
- **Not a medical device.** Triage + referral only.

**"What would you do next?"**
> "Full-data DistilBERT training; a clinically-labeled in-domain dataset;
> per-demographic fairness/error analysis before any real use; calibrated
> uncertainty on the crisis layer; and A/B-testing response strategies once I have
> outcome data."

---

## 8. Curveballs & behavioral

**"What was the hardest part?"**
> "Getting the crisis layer's operating point right. My first version was so
> recall-hungry it flagged 'that got me down today' as a crisis — which erodes
> trust. Diagnosing that the ML detector is unreliable on short text and
> redesigning the safety decision as lexicon-OR-high-confidence was the key
> insight."

**"What would you change if you started over?"**
> "Source an in-domain conversational dataset from day one instead of adapting a
> long-post corpus — most of my hard problems came from that mismatch."

**"What are you most proud of?"**
> "That every design choice is backed by a measurement or a stated principle.
> There's no magic number I can't justify — including the decision to *remove*
> complexity where the data didn't support it."

**"Did you use AI to build it?"**
> "Yes, as a pair-programmer — but I made the architecture decisions, chose the
> asymmetric-cost framing, designed the evaluation, and can defend every line.
> The measurements are real and reproducible."

---

## 9. One-liners to keep in your pocket

- "Errors are asymmetric, so I decoupled the safety objective from the accuracy objective."
- "I let the data choose the architecture — the ablation is the justification."
- "The cascade is a cost optimizer: transformer accuracy at a fraction of transformer cost."
- "Recall-first for crisis; macro-F1 for routine; nothing is a black box."
- "It degrades gracefully — safety never depends on an external API."

---

## 10. Know these exact numbers cold

| Thing | Number |
|---|---|
| Dataset size / classes | ~51k / 7 |
| Cascade accuracy / macro-F1 | 0.791 / 0.748 |
| SVC alone / DistilBERT alone (F1) | 0.716 / 0.731 |
| Cascade median latency vs BERT | ~3.5 ms vs ~59 ms (~17×) |
| % escalated to DistilBERT | ~42% |
| Crisis recall (deployed recall-first capability) | 95.7% |
| Crisis recall if F1-tuned instead | 79.9% |
