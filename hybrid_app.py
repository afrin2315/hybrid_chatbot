"""Hybrid mental-health triage chatbot -- Flask backend.

Architecture (see REPORT.md for the evaluation that justifies it):

    message
       |
       v  ml.cascade.Cascade
    [ safety layer ]  high-recall crisis detector, always on  -> CRISIS override
    [ fast tier   ]  calibrated TF-IDF + LinearSVC, ~1ms       -> confident cases
    [ accurate tier] DistilBERT, only for low-confidence cases -> hard cases
       |
       v  responder.generate  (Gemini if configured, else local templates)
    empathetic reply grounded by the real classifier label

There is no hand-mocked classification anywhere: every emotional tag comes from
a trained model, and the response layer only chooses *how* to reply.
"""
import os
import secrets
import sqlite3
import time


def _load_dotenv(path):
    """Minimal .env loader (no dependency). Lets you keep API keys in a
    gitignored .env instead of exporting them each run. Must run before the
    modules that read those keys are imported."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask import session as flask_session
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

import responder

# --- App setup ---
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
_cookie_secure = str(os.environ.get("SESSION_COOKIE_SECURE", "")).lower() in ("1", "true", "yes")
_cookie_samesite = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")
# Browsers reject a SameSite=None cookie that isn't also Secure (which requires
# HTTPS). Locally (HTTP) that silently drops the session and login "fails".
# So only honor None when Secure is on; otherwise fall back to Lax.
if _cookie_samesite.lower() == "none" and not _cookie_secure:
    print("SESSION_COOKIE_SAMESITE=None requires HTTPS; falling back to 'Lax' "
          "for this (insecure) context.")
    _cookie_samesite = "Lax"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=_cookie_samesite,
    SESSION_COOKIE_SECURE=_cookie_secure,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "templates")
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT_DIR, "app.db"))
HISTORY_TURNS = 12  # how many past messages to feed the responder

# --- Lazy cascade load (so auth/pages work even if artifacts are missing) ---
_cascade = None
_cascade_error = None


def get_cascade():
    global _cascade, _cascade_error
    if _cascade is None and _cascade_error is None:
        try:
            from ml.cascade import Cascade
            _cascade = Cascade()
        except Exception as e:  # artifacts not trained yet
            _cascade_error = str(e)
            print(f"Cascade unavailable ({e}). "
                  f"Run: python -m ml.data && python -m ml.train_svc && "
                  f"python -m ml.crisis")
    return _cascade


# --- DB ---
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 email TEXT NOT NULL UNIQUE,
                 password_hash TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')))"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 role TEXT NOT NULL,
                 text TEXT NOT NULL,
                 label TEXT,
                 confidence REAL,
                 tier TEXT,
                 crisis INTEGER DEFAULT 0,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')))"""
        )
        conn.commit()
    finally:
        conn.close()


_db_init()


def _uid():
    return flask_session.get("user_id")


def _require_login():
    uid = _uid()
    if not uid:
        return None, (jsonify({"error": "Not authenticated"}), 401)
    return uid, None


def _load_history(uid, limit=HISTORY_TURNS):
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT role, text FROM messages WHERE user_id=? "
            "ORDER BY id DESC LIMIT ?", (uid, limit)).fetchall()
    finally:
        conn.close()
    return [{"type": r["role"], "text": r["text"]} for r in reversed(rows)]


def _save_message(uid, role, text, pred=None):
    conn = _db()
    try:
        conn.execute(
            "INSERT INTO messages (user_id, role, text, label, confidence, "
            "tier, crisis) VALUES (?,?,?,?,?,?,?)",
            (uid, role, text,
             pred.label if pred else None,
             pred.confidence if pred else None,
             pred.tier if pred else None,
             1 if (pred and pred.crisis) else 0))
        conn.commit()
    finally:
        conn.close()


# --- Pages ---
@app.route("/", methods=["GET"])
def root():
    # Public landing page; the "try it" CTAs lead to /login and /app.
    return send_from_directory(FRONTEND_DIR, "landing.html")


@app.route("/login", methods=["GET"])
def login_page():
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/app", methods=["GET"])
def app_page():
    if not _uid():
        return redirect("/login")
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cascade_loaded": get_cascade() is not None}), 200


# --- Auth ---
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400
    conn = _db()
    try:
        cur = conn.execute("INSERT INTO users (email, password_hash) VALUES (?,?)",
                           (email, generate_password_hash(password)))
        conn.commit()
        uid = str(cur.lastrowid)
    except sqlite3.IntegrityError:
        return jsonify({"error": "account already exists"}), 409
    finally:
        conn.close()
    flask_session["user_id"] = uid
    return jsonify({"ok": True, "user": {"id": uid, "email": email}}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    conn = _db()
    try:
        row = conn.execute("SELECT id, email, password_hash FROM users "
                           "WHERE email=?", (email,)).fetchone()
    finally:
        conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid credentials"}), 401
    flask_session["user_id"] = str(row["id"])
    return jsonify({"ok": True, "user": {"id": str(row["id"]),
                                         "email": row["email"]}}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    flask_session.pop("user_id", None)
    return jsonify({"ok": True}), 200


@app.route("/api/me", methods=["GET"])
def me():
    uid = _uid()
    if not uid:
        return jsonify({"authenticated": False}), 200
    conn = _db()
    try:
        row = conn.execute("SELECT id, email FROM users WHERE id=?",
                           (uid,)).fetchone()
    finally:
        conn.close()
    if not row:
        flask_session.pop("user_id", None)
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True,
                    "user": {"id": str(row["id"]), "email": row["email"]}}), 200


@app.route("/api/history", methods=["GET"])
def history():
    uid, err = _require_login()
    if err:
        return err
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT role, text, label, confidence, tier, crisis, created_at "
            "FROM messages WHERE user_id=? ORDER BY id ASC", (uid,)).fetchall()
    finally:
        conn.close()
    return jsonify({"messages": [dict(r) for r in rows]}), 200


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    uid, err = _require_login()
    if err:
        return err
    conn = _db()
    try:
        conn.execute("DELETE FROM messages WHERE user_id=?", (uid,))
        conn.commit()
    finally:
        conn.close()
    return jsonify({"ok": True}), 200


# --- Chat (the real hybrid pipeline) ---
@app.route("/chat", methods=["POST"])
def chat():
    uid, err = _require_login()
    if err:
        return err
    if not request.json or "message" not in request.json:
        return jsonify({"error": "missing 'message'"}), 400
    user_message = str(request.json["message"]).strip()
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    cascade = get_cascade()
    if cascade is None:
        return jsonify({"error": "Model artifacts not available. Train models "
                                 "first (see README)."}), 503

    history = _load_history(uid)
    # DISABLE_ACCURATE_TIER lets us serve on the fast + safety tiers only (e.g.
    # while DistilBERT is being retrained, or for a lightweight deployment).
    use_accurate = os.environ.get("DISABLE_ACCURATE_TIER", "").lower() not in ("1", "true", "yes")
    pred = cascade.predict(user_message, use_accurate=use_accurate)
    reply = responder.generate(user_message, pred.label, history, pred.crisis)

    _save_message(uid, "user", user_message, pred)
    _save_message(uid, "model", reply)

    # Transparency: expose the full basis for the decision (no black box).
    top_probs = dict(sorted(pred.probs.items(), key=lambda kv: kv[1],
                            reverse=True)[:4]) if pred.probs else {}
    return jsonify({
        "reply": reply,
        "emotion": pred.label,
        "confidence": round(pred.confidence, 4),
        "suggestions": responder.suggestions_for(pred.label),
        "ts": time.time(),
        "explain": {
            "decided_by": pred.tier,           # safety | fast | accurate
            "crisis": pred.crisis,
            "crisis_score": round(pred.crisis_score, 4),
            "class_probabilities": {k: round(v, 4) for k, v in top_probs.items()},
        },
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f" * Frontend: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)
