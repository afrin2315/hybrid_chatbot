FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOST=0.0.0.0
ENV PORT=8080
ENV SESSION_COOKIE_SECURE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Lean serving image: fast + safety tiers (scikit-learn) + LLM replies. No torch.
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Bind to $PORT so the same image works on Cloud Run / Railway / Fly / Render.
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120"]
