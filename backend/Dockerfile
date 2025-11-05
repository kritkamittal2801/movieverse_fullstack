FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --only-binary=:all: --prefer-binary || pip install -r requirements.txt --no-cache-dir

COPY . .

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 10000

CMD gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app




