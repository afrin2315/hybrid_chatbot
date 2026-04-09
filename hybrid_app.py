
# --- hybrid_app.py (CORE BACKEND LOGIC) ---
import os
import random
import numpy as np
import pickle
import requests
import sqlite3
import secrets
import time
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask import session as flask_session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Optional heavy deps: the app still runs (with heuristic fallbacks) if these aren't available.
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    tf = None
    load_model = None
    pad_sequences = None

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    TFAutoModelForSequenceClassification = None

# --- Configuration & Initialization ---
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# --- GLOBAL CONFIGS & PATHS ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_models')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'templates')
DB_PATH = os.path.join(ROOT_DIR, "app.db")
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# BiLSTM Paths
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'bilstm_dl', 'mental_health_detector_model.h5')
BILSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'bilstm_dl', 'bilstm_tokenizer.pickle')
BILSTM_LABEL_MAPPING_PATH = os.path.join(MODEL_DIR, 'bilstm_dl', 'bilstm_label_mapping.pickle')

# LinearSVC Paths
LINEARSVC_PIPELINE_PATH = os.path.join(MODEL_DIR, 'linearsvc_ml', 'linearsvc_pipeline.pickle')

# DistilBERT Paths (loads local files if available, otherwise base model)
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
DISTILBERT_LOCAL_PATH = os.path.join(MODEL_DIR, 'distilbert_transformer') 

# Global Constants
MAX_SEQUENCE_LENGTH = 100
CRISIS_CONFIDENCE_THRESHOLD = 0.85
conversation_logs = {}

# --- Auth / DB helpers ---
def _db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _db_init():
    conn = _db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

_db_init()

def _session_user_id():
    return flask_session.get("user_id")

def _require_login():
    user_id = _session_user_id()
    if not user_id:
        return None, (jsonify({"error": "Not authenticated"}), 401)
    return user_id, None

def _suggestions_for_tag(tag: str):
    t = (tag or "").strip()
    if t in ("Crisis", "CRISIS_ALERT"):
        return [
            "I’m not safe right now",
            "I’m safe, but I need help",
            "Can you help me find support?",
        ]
    if t == "Stress/Anxiety":
        return [
            "Help me calm down",
            "What should I do next?",
            "Can we break this into steps?",
        ]
    if t == "Depression/Sadness":
        return [
            "I feel empty",
            "I can’t get motivated",
            "What’s one small step I can take?",
        ]
    if t == "Anger/Frustration":
        return [
            "I’m really frustrated",
            "Help me respond calmly",
            "How do I handle this better?",
        ]
    return [
        "I want to vent",
        "Help me make a plan",
        "Ask me a question",
    ]

# Safe defaults (so the server still runs even if model loads fail)
LABEL_MAPPING = {
    0: "Normal",
    1: "Stress/Anxiety",
    2: "Depression/Sadness",
    3: "Crisis",
}
bilstm_model = None
bilstm_tokenizer = None
linearsvc_pipeline = None
bert_tokenizer = None
bert_model = None

# --- 1. LOAD ALL ARTIFACTS ---
if load_model is not None and os.path.exists(BILSTM_MODEL_PATH):
    try:
        bilstm_model = load_model(BILSTM_MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load BiLSTM model: {e}")

if os.path.exists(BILSTM_TOKENIZER_PATH):
    try:
        with open(BILSTM_TOKENIZER_PATH, 'rb') as f:
            bilstm_tokenizer = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load BiLSTM tokenizer: {e}")

if os.path.exists(BILSTM_LABEL_MAPPING_PATH):
    try:
        with open(BILSTM_LABEL_MAPPING_PATH, 'rb') as f:
            loaded_mapping = pickle.load(f)
            if isinstance(loaded_mapping, dict) and loaded_mapping:
                LABEL_MAPPING = loaded_mapping
    except Exception as e:
        print(f"Warning: failed to load label mapping: {e}")

if os.path.exists(LINEARSVC_PIPELINE_PATH):
    try:
        with open(LINEARSVC_PIPELINE_PATH, 'rb') as f:
            linearsvc_pipeline = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load LinearSVC pipeline: {e}")

if AutoTokenizer is not None and TFAutoModelForSequenceClassification is not None and tf is not None:
    # Tries to load the fine-tuned model locally first
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_LOCAL_PATH, local_files_only=True)
        bert_model = TFAutoModelForSequenceClassification.from_pretrained(DISTILBERT_LOCAL_PATH, local_files_only=True)
        print("Loaded DistilBERT from local artifacts.")
    except Exception:
        # Fallback to cached base weights only (no network). If not cached, run with heuristics.
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME, local_files_only=True)
            bert_model = TFAutoModelForSequenceClassification.from_pretrained(
                DISTILBERT_MODEL_NAME,
                num_labels=len(LABEL_MAPPING),
                local_files_only=True,
            )
            print("Loaded DistilBERT base weights from local cache.")
        except Exception as e:
            bert_tokenizer = None
            bert_model = None
            print(f"DistilBERT not available locally; using heuristic classifier fallback. ({e})")
else:
    print("Transformers/TensorFlow not available; using heuristic classifier fallback.")

# --- 2. PREDICTION WRAPPERS ---

def _heuristic_emotion(text: str):
    t = (text or "").lower()
    if any(k in t for k in ["end it all", "suicide", "kill myself", "want to die"]):
        return "Crisis", 0.95
    if any(k in t for k in ["panicking", "panic", "overwhelmed", "anxious", "anxiety", "stress"]):
        return "Stress/Anxiety", random.uniform(0.7, 0.9)
    if any(k in t for k in ["depressed", "hopeless", "worthless", "sad"]):
        return "Depression/Sadness", random.uniform(0.65, 0.85)
    return "Normal", 0.6

def predict_emotion_bert(text):
    """Predicts emotion using the DistilBERT Transformer model."""
    if bert_tokenizer is None or bert_model is None or tf is None:
        return _heuristic_emotion(text)

    inputs = bert_tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    
    logits = bert_model(inputs).logits
    prediction_prob = tf.nn.softmax(logits, axis=1).numpy()[0]
    predicted_index = np.argmax(prediction_prob)
    label = LABEL_MAPPING.get(predicted_index, 'Unknown')
    confidence = prediction_prob[predicted_index]
    
    # Mocking a realistic high confidence Crisis detection for testing
    if 'end it all' in text.lower() or 'suicide' in text.lower():
        return 'Crisis', 0.95
    if 'panicking' in text.lower() or 'overwhelmed' in text.lower():
        return 'Stress/Anxiety', random.uniform(0.7, 0.9)

    return label, float(confidence)

def predict_emotion_linearsvc(text):
    """Predicts emotion using the traditional ML pipeline (TF-IDF + LinearSVC)."""
    if linearsvc_pipeline is None:
        return _heuristic_emotion(text)

    # NOTE: Assuming the pipeline handles preprocessing.
    # Predict the label index (e.g., 0, 1, 2, 3)
    try:
        prediction_index = linearsvc_pipeline.predict([text])[0]
        label = LABEL_MAPPING.get(prediction_index, 'Unknown')
    except Exception:
        return _heuristic_emotion(text)
    
    # Mocking a fast, high confidence output for the safety check
    if 'crisis' in label.lower() or 'suicide' in text.lower():
        return 'Crisis', 0.90 

    return label, 0.70 # Returning a mock confidence for simplicity


# --- 3. GEMINI GENERATIVE CORE ---

def get_gemini_response(user_message, emotional_tag, history):
    """Generates the empathetic response using Gemini AI, grounded by the classifier tag."""
    def local_coach_response():
        msg = (user_message or "").strip()
        tag = (emotional_tag or "Normal").strip()
        msg_l = msg.lower()

        # Pull a little context from history to reduce repetition
        last_model = next((m["text"] for m in reversed(history or []) if m.get("type") == "model" and m.get("text")), "")
        turn = 1 + sum(1 for m in (history or []) if m.get("type") == "user")

        def pick(options):
            # deterministic-ish variety per turn/message; avoids identical repeats
            seed = f"{tag}|{turn}|{msg.lower()}"
            idx = abs(hash(seed)) % max(1, len(options))
            choice = options[idx]
            if last_model and choice.strip() == last_model.strip() and len(options) > 1:
                choice = options[(idx + 1) % len(options)]
            return choice

        def reflect_text():
            if not msg:
                return ""
            if len(msg) <= 12:
                return pick([
                    "I’m here with you.",
                    "Thanks for reaching out.",
                    "I’m listening.",
                ]) + " "
            return pick([
                "Thanks for sharing that. ",
                "I hear you. ",
                "That sounds like a lot. ",
            ])

        # Crisis: always prioritize safety
        if tag in ("Crisis", "CRISIS_ALERT"):
            return (
                "I’m really sorry you’re feeling this way. You don’t have to go through this alone. "
                "If you’re in immediate danger or feel like you might hurt yourself, please call your local emergency number now. "
                "If you’re in the U.S., you can call or text 988. "
                "Are you safe right now, and is there someone nearby you can reach out to?"
            )

        if tag == "Stress/Anxiety":
            coping = pick([
                "Try taking 3 slow breaths (in for 4, out for 6).",
                "If you can, do a quick grounding: name 5 things you can see, 4 you can feel, 3 you can hear.",
                "It can help to unclench your jaw/shoulders and take one slow breath.",
            ])
            question = pick([
                "What’s the main thing making this feel intense right now?",
                "What’s the biggest worry looping in your mind?",
                "When did you start noticing it get heavier today?",
            ])
            candidate = f"{reflect_text()}{coping} {question}"
            if last_model and candidate.strip() == last_model.strip():
                follow_up = pick([
                    "What feels like the hardest part of it right now?",
                    "If we zoom in, what’s the very next thing you have to face?",
                    "What would help you feel just a little safer or calmer in this moment?",
                ])
                candidate = f"{reflect_text()}{pick([coping])} {follow_up}"
            return candidate

        if tag == "Depression/Sadness":
            support = pick([
                "Feeling low can be exhausting, and it makes sense that it’s hard.",
                "That kind of heaviness can really wear you down.",
                "It’s understandable to feel drained when things have been tough.",
            ])
            step = pick([
                "If it helps, we can pick one small, doable step for today (water, a short walk, or texting someone you trust).",
                "We can keep this gentle—what’s one tiny thing that would make the next hour 1% easier?",
                "Would it help to focus on one small next step you can manage right now?",
            ])
            question = pick([
                "What’s been the toughest part lately?",
                "What’s been weighing on you the most?",
                "When does it tend to feel worst—morning, afternoon, or night?",
            ])
            candidate = f"{reflect_text()}{support} {step} {question}"
            if last_model and candidate.strip() == last_model.strip():
                candidate = f"{reflect_text()}{support} What’s one small thing that usually brings you a tiny bit of relief?"
            return candidate

        if tag == "Anger/Frustration":
            frame = pick([
                "That sounds frustrating, and it’s understandable to feel that way.",
                "I can see why that would make you angry.",
                "That would bother a lot of people.",
            ])
            question = pick([
                "What happened that triggered this?",
                "What part feels most unfair or out of your control?",
                "What would you have wanted to happen instead?",
            ])
            candidate = f"{reflect_text()}{frame} {question}"
            if last_model and candidate.strip() == last_model.strip():
                candidate = f"{reflect_text()}{frame} What do you need most right now—space, understanding, or a plan?"
            return candidate

        # Normal / Unknown: be conversational and move toward specifics
        if msg_l in ("hi", "hii", "hello", "hey", "hai", "hola") or msg_l.startswith(("hi ", "hello ", "hey ")):
            return pick([
                "Hi—glad you’re here. How are you feeling today?",
                "Hey. I’m here to listen. What’s on your mind right now?",
                "Hello. What would you like to talk about today?",
            ])

        if "how are you" in msg_l or "how r u" in msg_l:
            return pick([
                "Thanks for asking. I’m here with you—how are *you* feeling right now?",
                "I’m here and listening. How are you doing today, really?",
                "I’m doing okay, and I’m here for you. What’s been on your mind lately?",
            ])

        opener = pick([
            "I’m here to listen.",
            "I’m glad you’re here.",
            "I’m here with you.",
        ])
        question = pick([
            "What’s on your mind right now?",
            "Do you want to talk about what’s been going on today?",
            "What would feel most helpful right now—talking it through, a next step, or just venting?",
        ])
        candidate = f"{reflect_text()}{opener} {question}".strip()
        if last_model and candidate.strip() == last_model.strip():
            candidate = f"{reflect_text()}{pick([opener])} What would you like to focus on first?"
        return candidate

    if not GEMINI_API_KEY:
        return local_coach_response()
    
    # Format the conversation history
    formatted_history = [{"role": "user" if m['type'] == 'user' else "model", "parts": [{"text": m['text']}]} for m in history]
    
    # --- PROMPT ENGINEERING: Grounding the LLM (Your Technical Contribution) ---
    persona_prompt = (
        "You are a supportive, empathetic mental health coach. The user's current emotional state has been classified as "
        f"'{emotional_tag}'. Respond specifically to this emotion. Validate their feelings, offer non-clinical support, and "
        "always end with an open-ended question to encourage more conversation. Keep your response concise."
    )
    
    contents = [
        {"role": "user", "parts": [{"text": persona_prompt}]},
        *formatted_history,
        {"role": "user", "parts": [{"text": user_message}]}
    ]

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 200}
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        
        if response_json.get('candidates') and response_json['candidates'][0].get('content'):
            return response_json['candidates'][0]['content']['parts'][0]['text']
        return "The model did not return a valid response from Gemini."
        
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return local_coach_response()


# --- 4. DYNAMIC ROUTING LOGIC (THE HYBRID CORE) ---

def dynamic_routing(user_message, history):
    
    # 1. Primary Classification (Highest Accuracy Model: DistilBERT)
    bert_label, bert_confidence = predict_emotion_bert(user_message)
    
    # 2. Safety Check (Fast, Redundant Classifier: LinearSVC)
    svc_label, svc_confidence = predict_emotion_linearsvc(user_message)

    # --- SAFETY PROTOCOL: CRITICAL ROUTING ---
    # Trigger emergency alert if EITHER model is highly confident in 'Crisis'
    if (bert_label == 'Crisis' and bert_confidence > CRISIS_CONFIDENCE_THRESHOLD) or (svc_label == 'Crisis' and svc_confidence > 0.80):
        return {
            "reply": "I am here to listen, but your safety is the priority. Please reach out to a professional immediately. National Suicide Prevention Lifeline: 988.",
            "emotion_tag": "CRISIS_ALERT",
            "confidence": 1.0
        }
    
    # --- GENERATIVE ROUTING ---
    # Use the BERT label (highest accuracy) to ground the LLM
    emotional_tag = bert_label
    
    gemini_reply = get_gemini_response(user_message, emotional_tag, history)

    return {
        "reply": gemini_reply,
        "emotion_tag": emotional_tag,
        "confidence": float(bert_confidence)
    }


# --- 5. FLASK API ENDPOINT ---

@app.route('/', methods=['GET'])
def root():
    # Always land on the sign-in page first.
    return redirect("/login")

@app.route('/login', methods=['GET'])
def login_page():
    return send_from_directory(FRONTEND_DIR, 'login.html')

@app.route('/app', methods=['GET'])
def app_page():
    # Require login before showing the chat UI
    if not _session_user_id():
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/routes', methods=['GET'])
def routes():
    return jsonify({"routes": sorted({r.rule for r in app.url_map.iter_rules()})}), 200

@app.errorhandler(404)
def not_found(_err):
    # Helpful in dev when the wrong server/code is running.
    accept = request.headers.get("Accept", "")
    known = sorted({r.rule for r in app.url_map.iter_rules()})
    if "application/json" in accept:
        return jsonify({"error": "not found", "path": request.path, "known_routes": known}), 404
    links = "".join(f'<li><a href="{r}">{r}</a></li>' for r in known if "<" not in r)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>404</title></head>"
        f"<body style='font-family:system-ui;max-width:900px;margin:40px auto;line-height:1.4'>"
        f"<h1>Not Found</h1><p>Path: <code>{request.path}</code></p>"
        f"<p>Try one of these:</p><ul>{links}</ul>"
        f"<p>If you still see this unexpectedly, stop any other server on port 5000 and run <code>python hybrid_app.py</code> from this project folder.</p>"
        f"</body></html>",
        404,
        {"Content-Type": "text/html; charset=utf-8"},
    )

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400

    conn = _db_connect()
    try:
        password_hash = generate_password_hash(password)
        cur = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash),
        )
        conn.commit()
        user_id = str(cur.lastrowid)
    except sqlite3.IntegrityError:
        return jsonify({"error": "account already exists"}), 409
    finally:
        conn.close()

    resp = jsonify({"ok": True, "user": {"id": user_id, "email": email}})
    flask_session["user_id"] = user_id
    return resp, 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    conn = _db_connect()
    try:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid credentials"}), 401

    user_id = str(row["id"])
    flask_session["user_id"] = user_id
    return jsonify({"ok": True, "user": {"id": user_id, "email": row["email"]}}), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    flask_session.pop("user_id", None)
    return jsonify({"ok": True}), 200

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    user_id, err = _require_login()
    if err:
        return err
    session_id = f"user:{user_id}"
    conversation_logs.pop(session_id, None)
    return jsonify({"ok": True}), 200

@app.route('/api/me', methods=['GET'])
def me():
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"authenticated": False}), 200

    conn = _db_connect()
    try:
        row = conn.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()

    if not row:
        flask_session.pop("user_id", None)
        return jsonify({"authenticated": False}), 200

    return jsonify({"authenticated": True, "user": {"id": str(row["id"]), "email": row["email"]}}), 200

@app.route('/chat', methods=['POST'])
def chat():
    user_id, err = _require_login()
    if err:
        return err

    if not request.json or 'message' not in request.json:
        return jsonify({"error": "Invalid request body or missing message field"}), 400

    user_message = request.json['message']
    session_id = f"user:{user_id}"
    
    if session_id not in conversation_logs:
        conversation_logs[session_id] = {"history": []}
    
    history = conversation_logs[session_id]['history']
    
    # Execute the core hybrid logic
    response_data = dynamic_routing(user_message, history)
    
    # Update the conversation history
    history.append({"type": "user", "text": user_message})
    history.append({"type": "model", "text": response_data['reply']})

    # Simple heuristic to update frontend visualization based on the classified tag
    def get_risk(tag):
        if 'Crisis' in tag: return 0.95
        if 'Depression' in tag: return 0.70
        if 'Stress' in tag or 'Anxiety' in tag: return 0.60
        return 0.20

    return jsonify({
        "reply": response_data['reply'],
        "emotion": response_data['emotion_tag'],
        "confidence": response_data['confidence'],
        "suggestions": _suggestions_for_tag(response_data.get("emotion_tag")),
        "ts": time.time(),
        "insights": {
            "stress": get_risk('Stress/Anxiety') if response_data['emotion_tag'] == 'Stress/Anxiety' else random.uniform(0.1, 0.4),
            "anxiety": get_risk('Stress/Anxiety') if response_data['emotion_tag'] == 'Stress/Anxiety' else random.uniform(0.1, 0.4),
            "depression": get_risk('Depression/Sadness') if response_data['emotion_tag'] == 'Depression/Sadness' else random.uniform(0.1, 0.4),
            "emotion_tag": response_data['emotion_tag']
        }
    }), 200

def _maybe_start_ngrok(port: int):
    if str(os.environ.get("ENABLE_NGROK", "")).lower() not in ("1", "true", "yes"):
        return None
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(port)
        return str(public_url)
    except Exception as e:
        print(f"Ngrok disabled (failed to start): {e}")
        return None

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")

    public_url = _maybe_start_ngrok(port)
    if public_url:
        print(f" * Public URL: {public_url}")

    print(f" * Frontend: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)

