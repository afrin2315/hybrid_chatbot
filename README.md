---
title: Hybrid Mental Health Chatbot
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid AI mental health support chatbot with Flask, DistilBERT, LinearSVC, and Gemini integration.
---

# Hybrid Mental Health Chatbot

A hybrid AI mental health support chatbot that combines:
- DistilBERT for primary emotion classification
- LinearSVC for fast crisis-risk backup checks
- rule-based fallbacks for resilience
- Gemini for grounded supportive responses when an API key is available

This project includes user authentication, persistent chat sessions via SQLite, saved local ML artifacts, and a browser UI for demonstration.

## Why this is resume-worthy

This is stronger than a basic chatbot because it shows:
- hybrid AI system design
- multi-model routing for safety-aware inference
- full-stack integration with authentication and persistence
- graceful fallback behavior when cloud AI or heavy ML models are unavailable
- deployable product thinking, not just notebook experimentation

## Project structure

- `hybrid_app.py` - main Flask backend and routing logic
- `wsgi.py` - production WSGI entrypoint
- `templates/` - login and chat UI
- `saved_models/` - local ML artifacts
- `app.db` - SQLite user database

## Run locally

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Start the app:

```powershell
python hybrid_app.py
```

3. Open:

- `http://127.0.0.1:5000/`

## Environment variables

Optional:

- `GEMINI_API_KEY` - enables live Gemini responses
- `FLASK_SECRET_KEY` - fixed secret key for stable sessions across restarts
- `DB_PATH` - custom SQLite file path for deployment
- `HOST` - defaults to `127.0.0.1`
- `PORT` - defaults to `5000`
- `ENABLE_NGROK=1` - optional ngrok tunnel
- `SESSION_COOKIE_SECURE=1` - enable secure cookies in HTTPS deployments

## Deploy on Hugging Face Spaces

This repo is ready for a free Hugging Face Docker Space deployment.

### How it works

Hugging Face officially supports Spaces using `docker`, and Docker Spaces can host arbitrary web apps as long as the app listens on the configured port. This project uses `app_port: 7860` and a `Dockerfile`.

### Space setup steps

1. Create a new Space on Hugging Face.
2. Choose `Docker` as the SDK.
3. Upload or push this repository to the Space.
4. In the Space settings, add these secrets or variables:
   - `FLASK_SECRET_KEY`
   - `SESSION_COOKIE_SECURE=1`
   - `GEMINI_API_KEY` if you want live Gemini responses
5. Let the Space build automatically.

### Notes for free hosting

- Hugging Face free CPU Spaces are good for resume demos and project showcases.
- SQLite is fine for a demo, but Space storage is not ideal for long-term persistent user data.
- If the Space restarts, user accounts in `app.db` may not be durable unless you later move to an external database.

### If build memory is tight

- Your app already has graceful fallbacks for some model-loading failures.
- If TensorFlow is too heavy for the free tier, the app can still demonstrate the product flow, but model-backed behavior may be reduced.

## Demo flow

1. Open the login page
2. Create an account
3. Start chatting
4. Show how the app classifies emotion, routes through the hybrid pipeline, and responds safely in crisis scenarios

## Suggested resume wording

Built and deployed a hybrid AI mental health chatbot using Flask, DistilBERT, LinearSVC, SQLite, and Gemini API integration, with safety-aware routing, authentication, persistent sessions, and resilient local fallback logic.
