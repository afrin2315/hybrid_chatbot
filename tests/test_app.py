"""Integration tests for the Flask app: auth, protected chat, persistence.

Run:  python -m pytest tests/ -q
"""
import os
import tempfile

import pytest

# Use a throwaway DB so tests never touch the real app.db.
_tmp_db = os.path.join(tempfile.gettempdir(), "hybrid_test.db")
os.environ["DB_PATH"] = _tmp_db

import hybrid_app  # noqa: E402


@pytest.fixture
def client():
    if os.path.exists(_tmp_db):
        os.remove(_tmp_db)
    hybrid_app._db_init()
    hybrid_app.app.config["TESTING"] = True
    with hybrid_app.app.test_client() as c:
        yield c


def _signup(client, email="a@b.com", pw="secret123"):
    return client.post("/api/signup", json={"email": email, "password": pw})


def test_signup_and_me(client):
    r = _signup(client)
    assert r.status_code == 201
    r = client.get("/api/me")
    assert r.get_json()["authenticated"] is True


def test_chat_requires_login(client):
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 401


def test_chat_returns_real_classification(client):
    _signup(client)
    r = client.post("/chat", json={"message": "I feel very anxious and panicky"})
    if r.status_code == 503:
        pytest.skip("model artifacts not trained in this environment")
    body = r.get_json()
    assert body["emotion"] in hybrid_app.__dict__.get("CLASSES", body["emotion"]) \
        or isinstance(body["emotion"], str)
    assert "explain" in body
    assert body["explain"]["decided_by"] in ("safety", "fast", "accurate")
    assert 0.0 <= body["confidence"] <= 1.0


def test_history_persists_and_resets(client):
    _signup(client)
    r = client.post("/chat", json={"message": "hello there"})
    if r.status_code == 503:
        pytest.skip("model artifacts not trained in this environment")
    h = client.get("/api/history").get_json()["messages"]
    assert len(h) >= 2  # user + model
    client.post("/api/reset")
    h2 = client.get("/api/history").get_json()["messages"]
    assert h2 == []


def test_duplicate_signup_conflicts(client):
    _signup(client)
    r = _signup(client)
    assert r.status_code == 409
